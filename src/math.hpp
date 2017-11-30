#pragma once
#include <cmath>
#include "matrix.hpp"

namespace Math {
    /**
     * Takes a pixel value in [0, 255] and "normalizes" it. This is done by dividing
     * the pixel value by 255, multiplying it by 0.99 for scale, and then
     * adding 0.01 to shift the range to [0.01, 1.0].
     *
     * @param pixel The pixel value in range [0, 255].
     */
    inline double normalizePixel(const int pixel) {
        return (static_cast<double>(pixel) / 255.0) * 0.99 + 0.01;
    }

    /**
     * Calculates the percentage for a count out of a total. Returns their
     * quotient multiplied by 100.
     *
     * @param count The count.
     * @param total The total.
     * @return The percentage in range [0, 100].
     */
    inline double percentage(const size_t count, const size_t total) {
        return static_cast<double>(count) / static_cast<double>(total) * 100.0;
    }

    /**
     * Let f(x) = sigmoid(x). This function calculates either the sigmoid or
     * the first-order derivative of sigmoid for a real value`x`.
     *
     * @param x The value to calculate f(x) or f'(x).
     * @param derivative If false, return f(x). Else, return f'(x). By default, this is false.
     * @returns The result of f(x) or f'(x).
     */
    inline double sigmoid(const double x, const bool derivative = false) {
        if (derivative) return sigmoid(x) * (1 - sigmoid(x));
        return 1.0 / (1.0 + std::exp(-x));
    }

    /**
     * Applies the sigmoid or sigmoid derivative to each value in a matrix of
     * size `N * M`.
     *
     * @param matrix The matrix to apply sigmoid to.
     * @param derivative If false, f(x). Else, f'(x). By default, this is false.
     * @return A new matrix with sigmoid applied to all entries.
     */
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

