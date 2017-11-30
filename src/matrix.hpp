#pragma once
#include <array>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>

namespace Matrix {
    /**
     * Class representing a matrix of size `N*M` with entries of type `T`.
     *
     * @tparam T The Matrix entry type.
     * @tparam N The number of rows.
     * @tparam M The number of columns.
     */
    template<typename T, size_t N, size_t M>
    class Matrix {
    public:
        /**
         * Gets the row of the matrix located at index `idx`.
         *
         * @param idx The index of the row.
         * @throws std::out_of_range
         * @return The `idx`-th row.
         */
        std::array<T, M>& operator[](const size_t idx) {
            if (idx >= N) throw std::out_of_range{"`idx` is out of range!"};
            return _matrix[idx];
        }

        /**
         * Gets a constant reference to the row of the matrix.
         *
         * @param idx The index of the row.
         * @throws std::out_of_range
         * @return The `idx`-th row.
         */
        const std::array<T, M>& operator[](const size_t idx) const {
            if (idx >= N) throw std::out_of_range{"`idx` is out of range!"};
            return _matrix[idx];
        }

        /**
         * Transposes the current matrix. That is, for every entry `(i, j)`, we
         * swap it with its corresponding entry, `(j, i)`.
         *
         * @return The transpose of this matrix.
         */
        Matrix<T, M, N> transpose() const {
            Matrix<T, M, N> result{};

            for (size_t i = 0; i < N; ++i) {
                for (size_t j = 0; j < M; ++j) {
                    result[j][i] = _matrix[i][j];
                }
            }

            return result;
        }

        constexpr size_t rows() const noexcept {
            return N;
        }

        constexpr size_t cols() const noexcept {
            return M;
        }

    private:
        std::array<std::array<T, M>, N> _matrix;
    };

    /**
     * Prints the matrix in the following form:
     *   1  2  3  4
     *   5  6  7  8
     *   9 10 11 12
     *
     * A newline is not appened, therefore the user would have to add it
     * manually.
     *
     * @param out The output stream to print the matrix to.
     * @param matrix The matrix to print.
     */
    template<typename T, size_t N, size_t M>
    std::ostream& operator<<(std::ostream& out, const Matrix<T, N, M>& matrix) {
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < M; ++j) {
                out << matrix[i][j];
                if (j < M - 1) out << std::setw(5);
            }
            if (i < N - 1) out << std::endl;
        }

        return out;
    }

    /**
     * Subtracts a `matrix1` by `matrix2`. The two matrices must have the same
     * dimensions, otherwise there'd be a compile time error.
     *
     * @tparam T The Matrix entry type.
     * @tparam N The row count for the matrices.
     * @tparam M The column count for the matrices.
     * @param matrix1 The first matrix.
     * @param matrix2 The second matrix.
     * @return The result of `matrix1 - matrix2`.
     */
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

    /**
     * Multiplies a matrix `matrix` by a scalar value of the same entry type
     * `T`.
     *
     * @tparam T The Matrix entry type.
     * @tparam N The row count for the matrix.
     * @tparam M The column count for the matrix.
     * @param matrix The matrix.
     * @param scalar The scalar to multiply the each entry `(i, j)` by.
     * @return The result of `matrix1 - matrix2`.
     */
    template<typename T, size_t N, size_t M>
    Matrix<T, N, M> operator*(const Matrix<T, N, M>& matrix, const T& scalar) {
        Matrix<T, N, M> result{};

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < M; ++j) {
                result[i][j] = matrix[i][j] * scalar;
            }
        }

        return result;
    }

    /**
     * Multiplies the current matrix by another matrix. The two matrices must
     * be compatible in the sense that the columns this matrix must be equal to
     * the rows in the second matrix.
     *
     * @tparam T The Matrix entry type.
     * @tparam N The row count for the first matrix.
     * @tparam K The column count and row count for the first and second matrices, respectively.
     * @tparam M The column count for the second matrix.
     * @param matrix The matrix to multiply to.
     * @return A new matrix holding the matrix product.
     */
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
                for (size_t k = 0; k < K; ++k) {
                    result[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }

        return result;
    }

    /**
     * Constructs a matrix of size `N*M` with all values initialized to a
     * random real value between -1 and 1. If the weight at position `(i, j)`
     * is 0, then we increment that weight by 0.01.
     *
     * @tparam N The number of rows.
     * @tparam M The numbero f columns.
     * @return A new matrix with random values between -1 and 1.
     */
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
} // Matrix

