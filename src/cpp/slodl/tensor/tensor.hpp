#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <string>
#include <memory>

#include "tensor_storage.hpp"


class Tensor {
private:
    std::shared_ptr<TensorStorage> storage;
    std::size_t start_offset;
    std::vector<std::size_t> strides;
    std::vector<std::size_t> dims;

    static std::size_t get_size_for_dims(std::vector<std::size_t> dims);
    static std::vector<std::size_t> get_strides_for_dims(std::vector<std::size_t> dims);

    std::size_t get_offset_for_flat_index(std::size_t flat_index) const;

    Tensor(
        std::shared_ptr<TensorStorage> storage,
        std::size_t start_offset,
        std::vector<std::size_t> strides,
        std::vector<std::size_t> dims
    );
    
public:
    Tensor(std::vector<std::size_t> dims);
    Tensor(std::vector<std::size_t> dims, double init_value);
    Tensor(std::vector<std::size_t> dims, std::vector<double> data);
    
    const std::vector<std::size_t>& shape() const;

    const std::vector<std::size_t>& element_strides() const;

    double* data();
    const double* data() const;

    Tensor operator[](std::size_t index) const;

    Tensor(const Tensor& other) = default;
    Tensor& operator=(const Tensor& other);

    Tensor clone() const;

    double item() const;
    Tensor& operator=(double value);

    std::string repr() const;

    // const std::vector<double>& raw() const;
};

#endif
