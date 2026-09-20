#ifndef TENSOR_STORAGE_HPP
#define TENSOR_STORAGE_HPP

#include <vector>


class TensorStorage {
private:
    std::vector<double> data;

public:
    TensorStorage(std::size_t size, double init_value);
    explicit TensorStorage(std::vector<double> data);

    double* ptr() { return data.data(); }
    const double* ptr() const { return data.data(); }
    std::size_t size() const { return data.size(); }
};


#endif
