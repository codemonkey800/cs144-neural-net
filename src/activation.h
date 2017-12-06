#pragma once
#include "matrix.h"

/**
 * Class representing an activation function for a neural network. The
 * activation function is defined to be sigmoid, but any other activation
 * function could be implemented.
 */
class ActivationFunction {
public:
    template<size_t N, size_t M>
    Matrix::Matrix<double, N, M> activate(const Matrix::Matrix<double, N, M>& matrix, const bool derivative = false) const;

private:
    double sigmoid(const double x, const bool derivative = false) const;
};

