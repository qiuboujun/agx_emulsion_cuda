#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cstring>
#include <sys/stat.h>

// Include the same headers as the main plugin
#include "CameraLUTKernel.cuh"
#include "EmulsionKernel.cuh"
#include "DirCouplerKernel.cuh"
#include "DiffusionHalationKernel.cuh"
#include "GrainKernel.cuh"
#include "PaperKernel.cuh"

// Test data structure
struct TestPixel {
    float r, g, b;
    TestPixel(float r_, float g_, float b_) : r(r_), g(g_), b(b_) {}
};

// Load test data from file
std::vector<TestPixel> load_test_data(const std::string& filename) {
    std::vector<TestPixel> pixels;
    std::ifstream file(filename);
    std::string line;
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        float r, g, b;
        if (iss >> r >> g >> b) {
            pixels.emplace_back(r, g, b);
        }
    }
    
    return pixels;
}

// Test Camera LUT stage
void test_camera_lut_stage(const std::vector<TestPixel>& test_pixels) {
    std::cout << "=== C++ Camera LUT Stage Test ===" << std::endl;
    
    // Load the LUT (same as main plugin)
    const char* lutPath = "/usr/OFX/Plugins/AgXEmulsionPlugin.ofx.bundle/Contents/Linux-x86-64/data/exposure_lut.bin";
    std::cout << "Loading LUT from: " << lutPath << std::endl;
    
    FILE* f = fopen(lutPath, "rb");
    if (!f) {
        std::cerr << "ERROR: Could not open LUT file" << std::endl;
        return;
    }
    
    int W = 128, H = 128;
    size_t n = W * H * 3;
    std::vector<uint16_t> buf(n);
    size_t read = fread(buf.data(), sizeof(uint16_t), n, f);
    fclose(f);
    
    if (read != n) {
        std::cerr << "ERROR: Expected " << n << " values but read " << read << std::endl;
        return;
    }
    
    std::cout << "LUT loaded successfully" << std::endl;
    
    // Upload LUT to GPU
    UploadExposureLUTCUDA(buf.data(), W, H);
    
    // Upload ACES2065-1 RGB->XYZ matrix
    const float m[9] = {0.9525524f,0.0000000f,0.0000937f,
                        0.3439664f,0.7281661f,-0.0721325f,
                        0.0000000f,0.0000000f,1.0088252f};
    UploadCameraMatrixCUDA(m);
    
    std::cout << "LUT uploaded to GPU" << std::endl;
    
    // Process each test pixel
    for (size_t i = 0; i < test_pixels.size(); i++) {
        const auto& pixel = test_pixels[i];
        
        // Create input buffer (single pixel)
        float input_rgb[4] = {pixel.r, pixel.g, pixel.b, 1.0f};
        
        std::cout << "Processing pixel " << i << ": RGB=(" 
                  << pixel.r << "," << pixel.g << "," << pixel.b << ")" << std::endl;
        
        // Convert RGB to xy (same as CUDA kernel)
        float X = m[0]*pixel.r + m[1]*pixel.g + m[2]*pixel.b;
        float Y = m[3]*pixel.r + m[4]*pixel.g + m[5]*pixel.b;
        float Z = m[6]*pixel.r + m[7]*pixel.g + m[8]*pixel.b;
        float sum = X + Y + Z + 1e-8f;
        float x = X / sum;
        float y = Y / sum;
        
        std::cout << "  RGB->xy: (" << x << "," << y << ")" << std::endl;
        
        // Sample the LUT (simplified version of CUDA kernel)
        if (IsCameraLUTValid()) {
            // For testing, we'll use a simplified sampling approach
            // In the actual CUDA kernel, this would use tex2D
            float u = std::min(std::max(x, 0.0f), 1.0f);
            float v = std::min(std::max(y, 0.0f), 1.0f);
            
            // Convert to LUT indices
            int u_idx = u * (W - 1);
            int v_idx = v * (H - 1);
            
            // Sample the LUT (simplified - no bilinear interpolation for now)
            int idx = v_idx * W + u_idx;
            if (idx < n/3) {
                // Convert half to float
                uint16_t half_r = buf[idx*3+0];
                uint16_t half_g = buf[idx*3+1];
                uint16_t half_b = buf[idx*3+2];
                
                // Simple conversion (not exact half2float)
                float cmy_r = __half2float(*reinterpret_cast<const __half*>(&half_r));
                float cmy_g = __half2float(*reinterpret_cast<const __half*>(&half_g));
                float cmy_b = __half2float(*reinterpret_cast<const __half*>(&half_b));
                
                std::cout << "  Sampled CMY=(" << cmy_r << "," << cmy_g << "," << cmy_b << ")" << std::endl;
            } else {
                std::cout << "  ERROR: LUT index out of bounds" << std::endl;
            }
        } else {
            std::cout << "  ERROR: LUT not valid" << std::endl;
        }
        
        std::cout << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <test_data_file>" << std::endl;
        return 1;
    }
    
    // Load test data
    std::vector<TestPixel> test_pixels = load_test_data(argv[1]);
    if (test_pixels.empty()) {
        std::cerr << "ERROR: No test data loaded" << std::endl;
        return 1;
    }
    
    std::cout << "Loaded " << test_pixels.size() << " test pixels" << std::endl;
    
    // Test Camera LUT stage
    test_camera_lut_stage(test_pixels);
    
    return 0;
} 