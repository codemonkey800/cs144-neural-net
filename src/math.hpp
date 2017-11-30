#pragma once
#include <cmath>
#include "matrix.hpp"

namespace Math {
    inline double normalizePixel(const int pixel) {
        return (static_cast<double>(pixel) / 255.0) * 0.99 + 0.01;
    }

    inline double percentage(const size_t count, const size_t total) {
        return static_cast<double>(count) / static_cast<double>(total) * 100.0;
    }

    inline double sigmoid(const double x, const bool derivative = false) {
        if (derivative) return sigmoid(x) * (1 - sigmoid(x));
        return 1.0 / (1.0 + std::exp(-x));
    }

    template<size_t N, size_t M>
    Matrix::Matrix<double, N, M> sigmoid(
        const Matrix::Matrix<double, N, M>& matrix,
        const bool derivative = false
    ) {
        Matrix::Matrix<double, N, M> result{};

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < M; ++j) {
                result[i][j] = sigmoid(matrix[i][j], derivative);
            }
        }

        return result;
    }
} // Math

