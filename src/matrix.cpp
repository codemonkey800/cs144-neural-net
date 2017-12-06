#pragma once
#include <cmath>
#include <iomanip>
#include <random>
#include <stdexcept>
#include "matrix.h"

/**
 * Let f(x) = sigmoid(x). This function calculates either the sigmoid or
 * the first-order derivative of sigmoid for a real value`x`.
 *
 * @param x The value to calculate f(x) or f'(x).
 * @param derivative If false, return f(x). Else, return f'(x). By default, this is false.
 * @return The result of f(x) or f'(x).
 */
inline double sigmoid(const double x, const bool derivative = false) {
    if (derivative) return sigmoid(x) * (1 - sigmoid(x));
    return 1.0 / (1.0 + std::exp(-x));
}

namespace Matrix {
    template<typename T, size_t N, size_t M>
    std::array<T, M>& Matrix<T, N, M>::operator[](const size_t idx) {
        if (idx >= N) throw std::out_of_range{"`idx` is out of range!"};
        return _matrix[idx];
    }

    template<typename T, size_t N, size_t M>
    const std::array<T, M>& Matrix<T, N, M>::operator[](const size_t idx) const {
        if (idx >= N) throw std::out_of_range{"`idx` is out of range!"};
        return _matrix[idx];
    }

    template<typename T, size_t N, size_t M>
    Matrix<T, M, N> Matrix<T, N, M>::transpose() const {
        Matrix<T, M, N> result{};

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < M; ++j) {
                result[j][i] = _matrix[i][j];
            }
        }

        return result;
    }

    template<typename T, size_t N, size_t M>
    Matrix<T, N, M> operator-(const Matrix<T, N, M>& matrix1, const Matrix<T, N, M>& matrix2) {
        Matrix<T, N, M> result{};

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < M; ++j) {
                result[i][j] = matrix1[i][j] - matrix2[i][j];
            }
        }

        return result;
    }

    template<typename T, size_t N, size_t M>
    Matrix<T, N, M> operator^(const Matrix<T, N, M>& matrix1, const Matrix<T, N, M>& matrix2) {
        Matrix<T, N, M> result{};

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < M; ++j) {
                result[i][j] = matrix1[i][j] * matrix2[i][j];
            }
        }

        return result;
    }

    template<typename T, size_t N, size_t M>
    Matrix<T, N, M> operator*(const T& scalar, const Matrix<T, N, M>& matrix) {
        Matrix<T, N, M> result{};

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < M; ++j) result[i][j] = matrix[i][j] * scalar;

        }

        return result;
    }

    template<typename T, size_t N, size_t K, size_t M>
    Matrix<T, N, M> operator*(const Matrix<T, N, K>& matrix1, const Matrix<T, K, M>& matrix2) {
        Matrix<T, N, M> result{};

        // Standard O(n^3) algorithm for finding the matrix product. Here, we
        // iterate through the rows and columns of the matrix product.
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < M; ++j) {
                // Here, we're traversing the i-th row of matrix1 and the j-th
                // column of matrix2.  Since we're doing a dot product between
                // the row and column, we're continually adding the product of
                // the k-th element in the row and column.
                for (size_t k = 0; k < K; ++k) result[i][j] += matrix1[i][k] * matrix2[k][j];

            }
        }

        return result;
    }

    template<size_t N, size_t M>
    Matrix<double, N, M> operator-(const Matrix<double, N, M>& matrix) {
        return -1.0 * matrix;
    }

    template<size_t N, size_t M>
    Matrix<double, N, M> randomMatrix() {
        std::random_device rd;
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<> dis(-1, 1);

        Matrix<double, N, M> result{};
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < M; ++j) {
                result[i][j] = dis(gen);
                if (result[i][j] == 0) result[i][j] += 0.01;
            }
        }

        return result;
    }

    template<size_t N, size_t M>
    Matrix<double, N, M> sigmoid(const Matrix<double, N, M>& matrix, const bool derivative) {
        Matrix<double, N, M> result{};

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < M; ++j) {
                auto sig = 1.0 / (1.0 + std::exp(-matrix[i][j]));
                result[i][j] = derivative ? sig * (1 - sig) : sig;
            }
        }

        return result;
    }
} // Matrix

