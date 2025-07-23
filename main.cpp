#include "kernels.h" // Include the header for our filter
#include <iostream>

int main() {
    // Create a sample 5x5 matrix
    Matrix input(5, 5);
    input << 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 1, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0;

    std::cout << "--- Input Matrix ---" << std::endl;
    std::cout << input << std::endl << std::endl;

    double sigma = 1.0;
    // Call the CUDA function, which is now declared in the header.
    Matrix output = gaussian_filter(input, sigma);

    std::cout << "--- Output after CUDA Gaussian Filter (sigma=" << sigma << ", mode=reflect) ---" << std::endl;
    std::cout << output << std::endl;

    return 0;
}

