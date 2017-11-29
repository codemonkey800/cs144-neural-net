#include <iostream>
#include "matrix.hpp"

template<typename Arg>
void print(const Arg& arg) {
    std::cout << arg << std::endl;
}

int main() {
    auto mat = Matrix::randomMatrix<3, 2>();
    print(mat);
    print(mat.transpose());
}

