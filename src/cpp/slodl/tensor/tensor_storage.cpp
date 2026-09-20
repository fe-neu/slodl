#include <stdexcept>
#include <utility>

#include "tensor_storage.hpp"

TensorStorage::TensorStorage(std::size_t size, double init_value)
    : data(size, init_value) {}


TensorStorage::TensorStorage(std::vector<double> data)
    : data(std::move(data)) {}
