#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "slodl/tensor/tensor.hpp"
#include "slodl/tensor/tensor_storage.hpp"
#include "slodl/autograd/autograd.hpp"
#include "slodl/autograd/engine.hpp"
#include "slodl/tensor/shape.hpp"

Tensor::Tensor(std::vector<std::size_t> dims)
    : storage(std::make_shared<TensorStorage>(get_size_for_dims(dims), 0.0)),
    start_offset(0),
    strides(get_strides_for_dims(dims)),
    dims(dims),
    meta(std::make_shared<AutogradMeta>()) {}

Tensor::Tensor(std::vector<std::size_t> dims, double init_value)
    : storage(std::make_shared<TensorStorage>(get_size_for_dims(dims), init_value)),
    start_offset(0),
    strides(get_strides_for_dims(dims)),
    dims(dims),
    meta(std::make_shared<AutogradMeta>()) {}

Tensor::Tensor(std::vector<std::size_t> dims, std::vector<double> data)
    : start_offset(0),
    strides(get_strides_for_dims(dims)),
    dims(dims),
    meta(std::make_shared<AutogradMeta>()) {
        if(data.size() != get_size_for_dims(dims)) {
            throw std::out_of_range("Given Data does not fit given dimensions");
        }
        storage = std::make_shared<TensorStorage>(std::move(data));
    }

Tensor::Tensor(
    std::shared_ptr<TensorStorage> storage,
    std::size_t start_offset,
    std::vector<std::size_t> strides,
    std::vector<std::size_t> dims
) : storage(std::move(storage)),
    start_offset(start_offset),
    strides(std::move(strides)),
    dims(std::move(dims)),
    meta(std::make_shared<AutogradMeta>()) {}


std::size_t Tensor::get_size_for_dims(std::vector<std::size_t> dims) {
    std::size_t tensor_size = 1;
    for( const std::size_t& i : dims) {
        tensor_size *= i;
    }
    return tensor_size;
}

std::vector<std::size_t> Tensor::get_strides_for_dims(std::vector<std::size_t> dims) {
    std::vector<std::size_t> offsets(dims.size(), 1);
    for(std::size_t i = dims.size(); i-- > 1;){
        offsets[i - 1] = offsets[i] * dims[i];
    }
    return offsets;
}

std::size_t Tensor::get_offset_for_flat_index(std::size_t flat_index) const {
    std::size_t offset = start_offset;
    for(std::size_t i = dims.size(); i-- > 0;){
        offset += (flat_index % dims[i]) * strides[i];
        flat_index /= dims[i];
    }
    return offset;
}

const std::vector<std::size_t>& Tensor::shape() const { return dims; }

const std::vector<std::size_t>& Tensor::element_strides() const { return strides; }

double* Tensor::data() { return storage->ptr() + start_offset; }

const double* Tensor::data() const { return storage->ptr() + start_offset; }

Tensor Tensor::operator[](std::size_t index) const {
    if (dims.empty()) {
        throw std::out_of_range("Cannot index a 0-dimensional tensor");
    }
    if (index >= dims[0]) {
        throw std::out_of_range("Index out of range");
    }
    return Tensor(
        storage,
        start_offset + index * strides[0],
        std::vector<std::size_t>(strides.begin() + 1, strides.end()),
        std::vector<std::size_t>(dims.begin() + 1, dims.end())
    );
}

Tensor Tensor::expand(const std::vector<std::size_t>& shape) const {
    return Tensor(
        storage,
        start_offset,
        broadcast_strides(dims, strides, shape),
        shape
    );
}

Tensor Tensor::clone() const {
    const std::size_t element_count = get_size_for_dims(dims);
    std::vector<double> values(element_count);
    for(std::size_t i = 0; i < element_count; i++){
        values[i] = storage->ptr()[get_offset_for_flat_index(i)];
    }
    return Tensor(dims, std::move(values));
}

double Tensor::item() const {
    if (!dims.empty()) {
        throw std::out_of_range("item() requires a 0-dimensional tensor");
    }
    return storage->ptr()[start_offset];
}

Tensor& Tensor::copy_(const Tensor& other) {
    if (this == &other) {
        return *this;
    }
    if (dims != other.dims) {
        throw std::invalid_argument("Cannot assign a tensor of a different shape");
    }

    const std::size_t element_count = get_size_for_dims(dims);
    double* destination = storage->ptr();

    if (storage == other.storage) {
        std::vector<double> source_values(element_count);
        for(std::size_t i = 0; i < element_count; i++){
            source_values[i] = other.storage->ptr()[other.get_offset_for_flat_index(i)];
        }
        for(std::size_t i = 0; i < element_count; i++){
            destination[get_offset_for_flat_index(i)] = source_values[i];
        }
    }
    else {
        const double* source = other.storage->ptr();
        for(std::size_t i = 0; i < element_count; i++){
            destination[get_offset_for_flat_index(i)] = source[other.get_offset_for_flat_index(i)];
        }
    }
    return *this;
}

Tensor& Tensor::fill_(double value) {
    const std::size_t element_count = get_size_for_dims(dims);
    double* elements = storage->ptr();
    for(std::size_t i = 0; i < element_count; i++){
        elements[get_offset_for_flat_index(i)] = value;
    }
    return *this;
}

namespace {

constexpr std::size_t kSummaryThreshold = 1000;
constexpr std::size_t kEdgeItems = 3;

std::string format_value(double value) {
    std::ostringstream stream;
    stream << value;
    return stream.str();
}

void append_dimension(
    std::string& out,
    const double* data,
    const std::vector<std::size_t>& dims,
    const std::vector<std::size_t>& strides,
    std::size_t axis,
    std::size_t offset,
    bool summarize,
    std::size_t indent
) {
    if (axis == dims.size()) {
        out += format_value(data[offset]);
        return;
    }

    const std::size_t length = dims[axis];
    const bool is_innermost = (axis + 1 == dims.size());
    const bool skip_middle = summarize && length > 2 * kEdgeItems;

    const std::string separator =
        is_innermost ? ", " : ",\n" + std::string(indent + 1, ' ');

    out += "[";
    for (std::size_t i = 0; i < length; i++) {
        if (i > 0) {
            out += separator;
        }
        if (skip_middle && i == kEdgeItems) {
            out += "...";
            i = length - kEdgeItems - 1;
            continue;
        }
        append_dimension(out, data, dims, strides, axis + 1,
                         offset + i * strides[axis], summarize, indent + 1);
    }
    out += "]";
}

}

std::string Tensor::repr() const {
    const std::string prefix = "Tensor(";
    const std::size_t element_count = get_size_for_dims(dims);
    const bool summarize = element_count > kSummaryThreshold;

    std::string out = prefix;
    append_dimension(out, data(), dims, strides, 0, 0, summarize,
                     prefix.size());

    if (summarize) {
        out += ", shape=[";
        for (std::size_t i = 0; i < dims.size(); i++) {
            out += (i ? ", " : "") + std::to_string(dims[i]);
        }
        out += "]";
    }
    return out + ")";
}

bool Tensor::requires_grad() const {
    return meta->requires_grad;
}

Tensor& Tensor::requires_grad_(bool flag) {
    if (!is_leaf()) {
        throw std::invalid_argument(
            "Can only change requires_grad on a leaf tensor");
    }
    meta->requires_grad = flag;
    return *this;
}

bool Tensor::is_leaf() const
{
    return !meta->grad_fn;
}

const Tensor* Tensor::grad() const {
    return meta->grad.get();
}

void Tensor::backward() {
    if (!dims.empty()) {
        throw std::invalid_argument(
            "backward() requires a 0-dimensional tensor");
    }
    if (!requires_grad()) {
        throw std::invalid_argument(
            "backward() requires a tensor that requires a gradient");
    }

    // The starting gradient of a tensor with respect to itself is 1.
    const Tensor seed(dims, 1.0);

    // Via an edge rather than grad_fn directly, so that a scalar leaf
    // accumulates into its own grad instead of finding no graph at all.
    const Edge entry = gradient_edge(*this);
    run_backward(entry.node, seed);
}

Tensor Tensor::detach() const {
    return Tensor(
        storage,
        start_offset,
        strides,
        dims
    );
}

std::shared_ptr<AutogradMeta> Tensor::autograd_meta() const {
    return meta;
}
