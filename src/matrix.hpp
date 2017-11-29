#pragma once
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>

/**
 * Namespace containing the matrix class, a type alias for matrix rows, and a
 * matrix multiplication incompatible exception.
 */
namespace Matrix {
    using Row = double*;

    class IncompatibleMultiplicationError : public std::logic_error {
    public:
        IncompatibleMultiplicationError()
            : std::logic_error{"The matrices are incompatbile! Check the rows and columns."} {
        }
    };

    /**
     * Class representing a matrix of size N*M. The class is bare and doesn't
     * support all matrix operations. Rather, the API is kept to the minimum
     * required algorithms for the neural net.
     *
     * @tparam N The number of rows.
     * @tparam M The number of columns.
     */
    template<size_t N, size_t M>
    class Matrix {
    public:
        Matrix() {
            _matrix = new double*[N];
            for (size_t i = 0; i < N; ++i) {
                _matrix[i] = new double[M];
            }
        }

        ~Matrix() {
            for (size_t i = 0; i < N; ++i) {
                delete[] _matrix[i];
            }
            delete[] _matrix;
        }

        /**
         * Retrieves the row at position idx. If the value of idx is greater or
         * equal to the number of rows, then we throw an out_of_range error.
         *
         * @param idx The index of the row.
         * @return The row at index idx.
         * @throws std::out_of_range
         */
        Row operator[](const size_t idx) const {
            if (idx >= N) throw std::out_of_range{"Row index is out of range!"};
            return _matrix[idx];
        }

        /**
         * Transposes the current matrix. That is, for every entry (i, j), we
         * swap it with its corresponding entry, (j, i).
         *
         * @tparam M The rows of the result matrix, which is the columns of this matrix.
         * @tparam N The columns of the result matrix, which is the rows of this matrix.
         * @return The transpose of this matrix.
         */
        Matrix<M, N> transpose() const {
            Matrix<M, N> result;

            for (size_t i = 0; i < N; ++i) {
                for (size_t j = 0; j < M; ++j) {
                    result[j][i] = _matrix[i][j];
                }
            }

            return result;
        }

        /*
         * Accessor functions for the rows and cols.
         */
        size_t rows() const noexcept {
            return N;
        }

        size_t cols() const noexcept {
            return M;
        }

    private:
        double** _matrix;
    };

    /**
     * A column vector can be considered a matrix of N rows and 1 column. This
     * is used for the input when querying the network.
     */
    template<size_t N>
    using ColumnVector = Matrix<N, 1>;

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
    template<size_t N, size_t M>
    std::ostream& operator<<(std::ostream& out, const Matrix<N, M>& matrix) {
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < M; ++j) {
                out << matrix[i][j];
                if (j < M - 1) out << std::setw(10);
            }
            out << std::endl;
        }

        return out;
    }

    /**
     * Multiplies the current matrix by another matrix. The two matrices must
     * be compatible in the sense that the columns this matrix must be equal to
     * the rows in the second matrix. If they are not, the
     * Matrix::IncompatibleMultiplicationError will be thrown.
     *
     * @param matrix The matrix to multiply to.
     * @return A new matrix holding the matrix product.
     */
    template<size_t N, size_t K, size_t M>
    Matrix<N, M> operator*(const Matrix<N, K>& matrix1, const Matrix<K, M>& matrix2) {
        Matrix<N, M> result;

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
     * Constructs a matrix of size N*m with all values initialized to a random
     * value.
     */
    template<size_t N, size_t M>
    Matrix<N, M> randomMatrix() {
        std::random_device rd;
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<> dis(1, 5);

        Matrix<N, M> result;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) result[i][j] = dis(gen);
        }

        return result;
    }
} // Matrix

