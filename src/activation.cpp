#pragma once
#include <cmath>
#include "activation.h"

template<size_t N, size_t M>
Matrix::Matrix<double, N, M> ActivationFunction::activate(const Matrix::Matrix<double, N, M>& matrix, const bool derivative) const {
    Matrix::Matrix<double, N, M> result{};

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            result[i][j] = sigmoid(matrix[i][j], derivative);
        }
    }

    return result;
}

double ActivationFunction::sigmoid(const double x, const bool derivative) const {
    auto result = 1.0 / (1.0 + std::exp(-x));
    return derivative ? result * (1 - result) : result;
}

