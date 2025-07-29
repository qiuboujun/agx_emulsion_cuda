#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cstring>
#include <sys/stat.h>
#include <dlfcn.h>
#include <cmath>
#include <algorithm>
#include <limits>

// Include the same headers as the main plugin
#include "EmulsionKernel.cuh"
#include "CameraLUTKernel.cuh"
#include "DirCouplerKernel.cuh"
#include "DiffusionHalationKernel.cuh"
#include "GrainKernel.cuh"
#include "PaperKernel.cuh"

// Test data structure
struct TestPixel {
    float c, m, y;
    TestPixel(float c_, float m_, float y_) : c(c_), m(m_), y(y_) {}
};

// Load test data from file
std::vector<TestPixel> load_test_data(const std::string& filename) {
    std::vector<TestPixel> pixels;
    std::ifstream file(filename);
    std::string line;
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        float c, m, y;
        if (iss >> c >> m >> y) {
            pixels.emplace_back(c, m, y);
        }
    }
    
    return pixels;
}

// Check if a value is NaN
bool is_nan(float value) {
    return std::isnan(value);
}

// Load JSON profile and handle nan values
bool load_json_profile(const std::string& stock_name, std::vector<float>& log_exposure, 
                      std::vector<float>& density_r, std::vector<float>& density_g, std::vector<float>& density_b) {
    std::string json_path = "/usr/OFX/Plugins/data/profiles/" + stock_name + ".json";
    std::ifstream file(json_path);
    
    if (!file.is_open()) {
        std::cerr << "ERROR: Could not open JSON profile: " << json_path << std::endl;
        return false;
    }
    
    // Simple JSON parsing for the density curves
    std::string line;
    bool in_density_curves = false;
    bool in_log_exposure = false;
    std::vector<float> temp_log_exposure;
    std::vector<float> temp_density_r, temp_density_g, temp_density_b;
    
    while (std::getline(file, line)) {
        if (line.find("\"density_curves\"") != std::string::npos) {
            in_density_curves = true;
            continue;
        }
        if (line.find("\"log_exposure\"") != std::string::npos) {
            in_log_exposure = true;
            continue;
        }
        
        if (in_log_exposure) {
            if (line.find("]") != std::string::npos) {
                in_log_exposure = false;
                continue;
            }
            
            // Parse log_exposure values
            std::istringstream iss(line);
            std::string token;
            while (iss >> token) {
                if (token.find(",") != std::string::npos) {
                    token = token.substr(0, token.find(","));
                }
                if (token.find("[") != std::string::npos) {
                    token = token.substr(token.find("[") + 1);
                }
                if (token.find("]") != std::string::npos) {
                    token = token.substr(0, token.find("]"));
                }
                
                if (!token.empty() && token != "[" && token != "]") {
                    try {
                        float val = std::stof(token);
                        temp_log_exposure.push_back(val);
                    } catch (...) {
                        // Skip non-numeric values
                    }
                }
            }
        }
        
        if (in_density_curves) {
            if (line.find("]") != std::string::npos) {
                in_density_curves = false;
                continue;
            }
            
            // Parse density curve values (3 channels per line)
            std::istringstream iss(line);
            std::string token;
            std::vector<float> row_values;
            
            while (iss >> token) {
                if (token.find(",") != std::string::npos) {
                    token = token.substr(0, token.find(","));
                }
                if (token.find("[") != std::string::npos) {
                    token = token.substr(token.find("[") + 1);
                }
                if (token.find("]") != std::string::npos) {
                    token = token.substr(0, token.find("]"));
                }
                
                if (!token.empty() && token != "[" && token != "]") {
                    try {
                        if (token == "null" || token == "nan") {
                            row_values.push_back(std::numeric_limits<float>::quiet_NaN());
                        } else {
                            float val = std::stof(token);
                            row_values.push_back(val);
                        }
                    } catch (...) {
                        row_values.push_back(std::numeric_limits<float>::quiet_NaN());
                    }
                }
            }
            
            if (row_values.size() >= 3) {
                temp_density_r.push_back(row_values[0]);
                temp_density_g.push_back(row_values[1]);
                temp_density_b.push_back(row_values[2]);
            }
        }
    }
    
    // Filter out nan values (same as Python implementation)
    for (size_t i = 0; i < temp_log_exposure.size(); i++) {
        if (i < temp_density_r.size() && 
            !is_nan(temp_density_r[i]) && 
            !is_nan(temp_density_g[i]) && 
            !is_nan(temp_density_b[i])) {
            
            log_exposure.push_back(temp_log_exposure[i]);
            density_r.push_back(temp_density_r[i]);
            density_g.push_back(temp_density_g[i]);
            density_b.push_back(temp_density_b[i]);
        }
    }
    
    std::cout << "JSON profile loaded: " << log_exposure.size() << " valid samples (filtered from " 
              << temp_log_exposure.size() << " total)" << std::endl;
    
    return !log_exposure.empty();
}

// Load filtered CSV data (created by Python from JSON)
bool load_filtered_csv_data(const std::string& filename, std::vector<float>& log_exposure, 
                           std::vector<float>& density_r, std::vector<float>& density_g, std::vector<float>& density_b) {
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "ERROR: Could not open filtered CSV file: " << filename << std::endl;
        return false;
    }
    
    std::string line;
    // Skip header line
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        std::vector<float> values;
        
        while (std::getline(iss, token, ',')) {
            try {
                float val = std::stof(token);
                values.push_back(val);
            } catch (...) {
                // Skip invalid values
            }
        }
        
        if (values.size() >= 4) {
            log_exposure.push_back(values[0]);
            density_r.push_back(values[1]);
            density_g.push_back(values[2]);
            density_b.push_back(values[3]);
        }
    }
    
    std::cout << "Filtered CSV loaded: " << log_exposure.size() << " samples" << std::endl;
    return !log_exposure.empty();
}

// Test Emulsion stage
void test_emulsion_stage(const std::vector<TestPixel>& test_pixels) {
    std::cout << "=== C++ Emulsion Stage Test ===" << std::endl;
    
    // Load the film LUT (same as main plugin)
    const char* lutPath = "/usr/OFX/Plugins/data/film/negative/kodak_portra_400/";
    std::cout << "Loading film LUT from: " << lutPath << std::endl;
    
    // Load film LUT using the same method as main plugin
    void* handle = dlopen("/usr/OFX/Plugins/AgXEmulsionPlugin.ofx.bundle/Contents/Linux-x86-64/libAgXLUT.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "ERROR: Could not open libAgXLUT.so" << std::endl;
        return;
    }
    
    typedef bool (*LoadFilmLUTFunc)(const char*, float*, float*, float*, float*);
    LoadFilmLUTFunc loadFilmLUT = (LoadFilmLUTFunc)dlsym(handle, "loadFilmLUT");
    
    if (!loadFilmLUT) {
        std::cerr << "ERROR: Could not find loadFilmLUT function" << std::endl;
        dlclose(handle);
        return;
    }
    
    // Load film LUT
    float density_curves[3][1000];  // R, G, B curves
    float log_exposure[1000];
    
    bool result = loadFilmLUT("kodak_portra_400", log_exposure, density_curves[0], density_curves[1], density_curves[2]);
    
    if (!result) {
        std::cerr << "ERROR: Failed to load film LUT" << std::endl;
        dlclose(handle);
        return;
    }
    
    std::cout << "Film LUT loaded successfully" << std::endl;
    std::cout << "Curve sizes: R=" << 1000 << ", G=" << 1000 << ", B=" << 1000 << std::endl; // Curve sizes are not directly available from the new loadFilmLUT signature
    std::cout << "Exposure size: " << 1000 << std::endl; // Exposure size is not directly available from the new loadFilmLUT signature
    
    // Upload LUT to GPU
    UploadLUTCUDA(log_exposure, density_curves[0], density_curves[1], density_curves[2]);
    
    // Upload exposure EV (use default value)
    float exposureEV = 0.0f;
    // Note: Exposure EV is handled in the main kernel, not as a separate upload
    
    std::cout << "Film LUT uploaded to GPU" << std::endl;
    
    // Process each test pixel
    for (size_t i = 0; i < test_pixels.size(); i++) {
        const auto& pixel = test_pixels[i];
        
        // Create input buffer (single pixel)
        float input_cmy[4] = {pixel.c, pixel.m, pixel.y, 1.0f};
        
        std::cout << "Processing pixel " << i << ": CMY=(" 
                  << pixel.c << "," << pixel.m << "," << pixel.y << ")" << std::endl;
        
        // Convert CMY to log_raw (same as Python)
        float log_raw[3];
        for (int j = 0; j < 3; j++) {
            log_raw[j] = log10f(input_cmy[j] + 1e-10f);
        }
        
        std::cout << "  CMY->log_raw: (" << log_raw[0] << "," << log_raw[1] << "," << log_raw[2] << ")" << std::endl;
        
        // Apply gamma correction (same as Python)
        float gamma = 1.0f; // Default gamma value
        float gamma_corrected[3];
        for (int j = 0; j < 3; j++) {
            gamma_corrected[j] = log_raw[j] / gamma;
        }
        std::cout << "  Gamma corrected (log_raw / gamma): (" << gamma_corrected[0] << "," << gamma_corrected[1] << "," << gamma_corrected[2] << ")" << std::endl;
        
        // Apply emulsion processing (simplified version of CUDA kernel)
        // For testing, we'll use a simplified approach
        // In the actual CUDA kernel, this would use the uploaded LUT
        
        // Simple density curve lookup (simplified)
        float density[3];
        for (int j = 0; j < 3; j++) {
            // Find the closest exposure value in the LUT
            float min_diff = 1e10f;
            int best_idx = 0;
            
            // Print some sample LUT values for debugging
            if (i == 0 && j == 0) {
                std::cout << "  Sample LUT values (first 5):" << std::endl;
                for (int k = 0; k < 5; k++) {
                    std::cout << "    [" << log_exposure[k] << "] -> [" << density_curves[0][k] << ", " << density_curves[1][k] << ", " << density_curves[2][k] << "]" << std::endl;
                }
            }
            
            for (int k = 0; k < 1000; k++) { // Assuming exposure_size is 1000 for now
                float diff = fabsf(log_exposure[k] - gamma_corrected[j]);
                if (diff < min_diff) {
                    min_diff = diff;
                    best_idx = k;
                }
            }
            
            density[j] = density_curves[j][best_idx];
            
            // Print interpolation details for first pixel
            if (i == 0) {
                std::cout << "  Channel " << j << " interpolation:" << std::endl;
                std::cout << "    Target value: " << gamma_corrected[j] << std::endl;
                std::cout << "    Best match at index " << best_idx << ": " << log_exposure[best_idx] << std::endl;
                std::cout << "    Density value: " << density[j] << std::endl;
            }
        }
        
        std::cout << "  Density=(" << density[0] << "," << density[1] << "," << density[2] << ")" << std::endl;
        std::cout << std::endl;
    }
    
    dlclose(handle);
}

// Test Emulsion stage with JSON profile (nan filtering)
void test_emulsion_stage_json(const std::vector<TestPixel>& test_pixels) {
    std::cout << "=== C++ Emulsion Stage Test (JSON Profile) ===" << std::endl;
    
    // Load JSON profile with nan filtering
    std::vector<float> log_exposure, density_r, density_g, density_b;
    if (!load_json_profile("kodak_portra_400", log_exposure, density_r, density_g, density_b)) {
        std::cerr << "ERROR: Failed to load JSON profile" << std::endl;
        return;
    }
    
    std::cout << "JSON profile loaded successfully" << std::endl;
    std::cout << "Log exposure range: " << log_exposure.front() << " to " << log_exposure.back() << std::endl;
    std::cout << "Number of valid samples: " << log_exposure.size() << std::endl;
    std::cout << "Sample values:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(5), log_exposure.size()); i++) {
        std::cout << "  [" << log_exposure[i] << "] -> [" << density_r[i] << ", " << density_g[i] << ", " << density_b[i] << "]" << std::endl;
    }
    
    // Process each test pixel
    for (size_t i = 0; i < test_pixels.size(); i++) {
        const auto& pixel = test_pixels[i];
        
        std::cout << "Processing pixel " << i << ": CMY=(" 
                  << pixel.c << "," << pixel.m << "," << pixel.y << ")" << std::endl;
        
        // Convert CMY to log_raw (same as Python)
        float log_raw[3];
        for (int j = 0; j < 3; j++) {
            float input_cmy[3] = {pixel.c, pixel.m, pixel.y};
            log_raw[j] = log10f(input_cmy[j] + 1e-10f);
        }
        
        std::cout << "  CMY->log_raw: (" << log_raw[0] << "," << log_raw[1] << "," << log_raw[2] << ")" << std::endl;
        
        // Apply gamma correction (same as Python)
        float gamma = 1.0f; // Default gamma value
        float gamma_corrected[3];
        for (int j = 0; j < 3; j++) {
            gamma_corrected[j] = log_raw[j] / gamma;
        }
        std::cout << "  Gamma corrected (log_raw / gamma): (" << gamma_corrected[0] << "," << gamma_corrected[1] << "," << gamma_corrected[2] << ")" << std::endl;
        
        // Apply emulsion processing with JSON data
        float density[3];
        for (int j = 0; j < 3; j++) {
            // Find the closest exposure value in the LUT
            float min_diff = 1e10f;
            int best_idx = 0;
            
            for (size_t k = 0; k < log_exposure.size(); k++) {
                float diff = fabsf(log_exposure[k] - gamma_corrected[j]);
                if (diff < min_diff) {
                    min_diff = diff;
                    best_idx = k;
                }
            }
            
            // Get density value
            if (j == 0) {
                density[j] = density_r[best_idx];
            } else if (j == 1) {
                density[j] = density_g[best_idx];
            } else {
                density[j] = density_b[best_idx];
            }
            
            // Print interpolation details for first pixel
            if (i == 0) {
                std::cout << "  Channel " << j << " interpolation:" << std::endl;
                std::cout << "    Target value: " << gamma_corrected[j] << std::endl;
                std::cout << "    Best match at index " << best_idx << ": " << log_exposure[best_idx] << std::endl;
                std::cout << "    Density value: " << density[j] << std::endl;
            }
        }
        
        std::cout << "  Density=(" << density[0] << "," << density[1] << "," << density[2] << ")" << std::endl;
        std::cout << std::endl;
    }
}

// Test Emulsion stage with filtered CSV data
void test_emulsion_stage_filtered_csv(const std::vector<TestPixel>& test_pixels) {
    std::cout << "=== C++ Emulsion Stage Test (Filtered CSV Data) ===" << std::endl;
    
    // Load filtered CSV data
    std::vector<float> log_exposure, density_r, density_g, density_b;
    if (!load_filtered_csv_data("../filtered_kodak_portra_400_data.csv", log_exposure, density_r, density_g, density_b)) {
        std::cerr << "ERROR: Failed to load filtered CSV data" << std::endl;
        return;
    }
    
    std::cout << "Filtered CSV data loaded successfully" << std::endl;
    std::cout << "Log exposure range: " << log_exposure.front() << " to " << log_exposure.back() << std::endl;
    std::cout << "Number of valid samples: " << log_exposure.size() << std::endl;
    std::cout << "Sample values:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(5), log_exposure.size()); i++) {
        std::cout << "  [" << log_exposure[i] << "] -> [" << density_r[i] << ", " << density_g[i] << ", " << density_b[i] << "]" << std::endl;
    }
    
    // Process each test pixel
    for (size_t i = 0; i < test_pixels.size(); i++) {
        const auto& pixel = test_pixels[i];
        
        std::cout << "Processing pixel " << i << ": CMY=(" 
                  << pixel.c << "," << pixel.m << "," << pixel.y << ")" << std::endl;
        
        // Convert CMY to log_raw (same as Python)
        float log_raw[3];
        for (int j = 0; j < 3; j++) {
            float input_cmy[3] = {pixel.c, pixel.m, pixel.y};
            log_raw[j] = log10f(input_cmy[j] + 1e-10f);
        }
        
        std::cout << "  CMY->log_raw: (" << log_raw[0] << "," << log_raw[1] << "," << log_raw[2] << ")" << std::endl;
        
        // Apply gamma correction (same as Python)
        float gamma = 1.0f; // Default gamma value
        float gamma_corrected[3];
        for (int j = 0; j < 3; j++) {
            gamma_corrected[j] = log_raw[j] / gamma;
        }
        std::cout << "  Gamma corrected (log_raw / gamma): (" << gamma_corrected[0] << "," << gamma_corrected[1] << "," << gamma_corrected[2] << ")" << std::endl;
        
        // Apply emulsion processing with filtered data
        float density[3];
        for (int j = 0; j < 3; j++) {
            // Find the closest exposure value in the LUT
            float min_diff = 1e10f;
            int best_idx = 0;
            
            for (size_t k = 0; k < log_exposure.size(); k++) {
                float diff = fabsf(log_exposure[k] - gamma_corrected[j]);
                if (diff < min_diff) {
                    min_diff = diff;
                    best_idx = k;
                }
            }
            
            // Get density value
            if (j == 0) {
                density[j] = density_r[best_idx];
            } else if (j == 1) {
                density[j] = density_g[best_idx];
            } else {
                density[j] = density_b[best_idx];
            }
            
            // Print interpolation details for first pixel
            if (i == 0) {
                std::cout << "  Channel " << j << " interpolation:" << std::endl;
                std::cout << "    Target value: " << gamma_corrected[j] << std::endl;
                std::cout << "    Best match at index " << best_idx << ": " << log_exposure[best_idx] << std::endl;
                std::cout << "    Density value: " << density[j] << std::endl;
            }
        }
        
        std::cout << "  Density=(" << density[0] << "," << density[1] << "," << density[2] << ")" << std::endl;
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
    
    // Test Emulsion stage with CSV data
    test_emulsion_stage(test_pixels);
    
    // Test Emulsion stage with JSON profile (nan filtering)
    test_emulsion_stage_json(test_pixels);

    // Test Emulsion stage with filtered CSV data
    test_emulsion_stage_filtered_csv(test_pixels);
    
    return 0;
} 