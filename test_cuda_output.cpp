#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <dlfcn.h>

// Function pointers for CUDA kernels (using actual exported names)
typedef void (*LaunchDirCouplerCUDA_t)(float*, int, int);
typedef void (*PaperKernel_t)(float*, int, int);

int main() {
    // Load the OFX plugin directly
    void* handle = dlopen("/usr/OFX/Plugins/AgXEmulsionPlugin.ofx.bundle/Contents/Linux-x86-64/AgXEmulsionPlugin.ofx", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load OFX plugin: " << dlerror() << std::endl;
        return 1;
    }
    
    // Get function pointers for available functions
    LaunchDirCouplerCUDA_t LaunchDirCouplerCUDA = (LaunchDirCouplerCUDA_t)dlsym(handle, "LaunchDirCouplerCUDA");
    PaperKernel_t PaperKernel = (PaperKernel_t)dlsym(handle, "_Z11PaperKernelPfii");
    
    if (!LaunchDirCouplerCUDA || !PaperKernel) {
        std::cerr << "Failed to get function pointers: " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }
    
    // Test data
    int width = 1, height = 1;
    std::vector<float> img(width * height * 4);
    
    // Initialize with test RGB [0.5, 0.3, 0.7]
    img[0] = 0.5f;  // R
    img[1] = 0.3f;  // G
    img[2] = 0.7f;  // B
    img[3] = 1.0f;  // A
    
    // Allocate GPU memory
    float* d_img;
    cudaMalloc(&d_img, img.size() * sizeof(float));
    cudaMemcpy(d_img, img.data(), img.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    std::cout << "CUDA KERNEL OUTPUTS (Available Functions):" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // Stage 3: DIR Coupler (available)
    LaunchDirCouplerCUDA(d_img, width, height);
    cudaMemcpy(img.data(), d_img, img.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "DIR Coupler: [" << img[0] << ", " << img[1] << ", " << img[2] << "]" << std::endl;
    
    // Stage 6: Paper (available)
    PaperKernel(d_img, width, height);
    cudaMemcpy(img.data(), d_img, img.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Paper: [" << img[0] << ", " << img[1] << ", " << img[2] << "]" << std::endl;
    
    // Cleanup
    cudaFree(d_img);
    dlclose(handle);
    
    return 0;
} 