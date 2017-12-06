#pragma once
#include <array>

namespace Matrix {
    /**
     * Class representing a matrix of size `N * M` with entries of type `T`.
     * All of the matrix calculations are immutable and therefore create new
     * matrix instances containing the results of the calculation.
     *
     * For example, multiplying a `N * K` matrix by a `K * M` matrix will
     * result in a completely new matrix of size `N * M`, with the original
     * matrices being left untouched.
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
        std::array<T, M>& operator[](const size_t idx);

        /**
         * Gets a constant reference to the row of the matrix.
         *
         * @param idx The index of the row.
         * @throws std::out_of_range
         * @return The `idx`-th row.
         */
        const std::array<T, M>& operator[](const size_t idx) const;

        /**
         * Transposes the current matrix. That is, for every entry `(i, j)`, we
         * swap it with its corresponding entry, `(j, i)`.
         *
         * @return The transpose of this matrix.
         */
        Matrix<T, M, N> transpose() const;

    private:
        std::array<std::array<T, M>, N> _matrix;
    };

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
    Matrix<T, N, M> operator-(const Matrix<T, N, M>& matrix1, const Matrix<T, N, M>& matrix2);

    /**
     * Calculates the Hadamard product of two matrices. That is, `A ^ B` for
     * matrices `A` and `B` will result in an element-wise matrix product of
     * size `N * M`. Conventially, the Hadamard product is represented with an
     * empty circle, but in the context of C++, we'll use ^.
     *
     * @tparam T The Matrix entry type.
     * @tparam N The row count for the matrices.
     * @tparam M The column count for the matrices.
     * @param matrix1 The first matrix.
     * @param matrix2 The second matrix.
     * @return The Hadamard product of `matrix1` and `matrix2`.
     */
    template<typename T, size_t N, size_t M>
    Matrix<T, N, M> operator^(const Matrix<T, N, M>& matrix1, const Matrix<T, N, M>& matrix2);

    /**
     * Multiplies a scalar value of Matrix entry type `T` to each entry in the
     * matrix.
     *
     * @tparam T The Matrix entry type.
     * @tparam N The row count for the matrix.
     * @tparam M The column count for the matrix.
     * @param scalar The scalar to multiply the each entry `(i, j)` by.
     * @param matrix The matrix.
     * @return The result of `matrix1 - matrix2`.
     */
    template<typename T, size_t N, size_t M>
    Matrix<T, N, M> operator*(const T& scalar, const Matrix<T, N, M>& matrix);

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
    Matrix<T, N, M> operator*(const Matrix<T, N, K>& matrix1, const Matrix<T, K, M>& matrix2);

    /**
     * Negates the matrix. Multiplies every entry by -1.
     *
     * @tparam N The number of rows.
     * @tparam M The number of columns.
     * @return A new matrix which is the original matrix but with all negated values.
     */
    template<size_t N, size_t M>
    Matrix<double, N, M> operator-(const Matrix<double, N, M>& matrix);

    /**
     * Constructs a matrix of size `N * M` with all values initialized to a
     * random real value between -1 and 1. If the weight at position `(i, j)`
     * is 0, then we increment that weight by 0.01.
     *
     * @tparam N The number of rows.
     * @tparam M The numbero f columns.
     * @return A new matrix with random values between -1 and 1.
     */
    template<size_t N, size_t M>
    Matrix<double, N, M> randomMatrix();
} // Matrix

