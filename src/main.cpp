#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "matrix.hpp"

template<typename Arg>
void print(const Arg& arg) {
    std::cout << arg << std::endl;
}

int main() {
    auto mat = Matrix::randomMatrix<2, 3>();
    print(mat);
    print(mat.transpose());
    print(mat * mat.transpose());
}

