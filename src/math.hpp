#pragma once
#include <cmath>
#include "matrix.hpp"

namespace Math {
    inline double sigmoid(const double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    template<size_t N, size_t M>
    Matrix::Matrix<double, N, M> sigmoid(const Matrix::Matrix<double, N, M>& matrix) {
        Matrix::Matrix<double, N, M> result;

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                result[i][j] = sigmoid(matrix[i][j]);
            }
        }

        return result;
    }
} // Math

