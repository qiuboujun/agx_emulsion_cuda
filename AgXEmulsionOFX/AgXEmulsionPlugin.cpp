#include "AgXEmulsionPlugin.h"

#include <stdio.h>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"
#include "ofxsLog.h"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstring>
#include "DynamicSpectralUpsampling.cuh"
#include "EmulsionKernel.cuh"
#include "DiffusionHalationKernel.cuh"
#include "GrainKernel.cuh"
#include "PaperKernel.cuh"
#include "DirCouplerKernel.cuh"
#include "CIE1931.cuh"
#include "couplers.hpp" // NEW include for DIR matrix computation
#include <cstdint>
#include <sys/stat.h> // Required for stat()
#include <cmath> // Required for std::isnan
#include <limits> // Required for std::numeric_limits
#include <nlohmann/json.hpp>
using json = nlohmann::json;

// Forward CUDA uploads for spectral stage
extern "C" bool UploadCameraSensCUDA(const float* sensR,const float* sensG,const float* sensB,int count);
extern "C" bool UploadSpectralLUTCUDA(const float* lutData,int w,int h,int samples);

#define kPluginName "AgX Emulsion"
#define kPluginGrouping "OpenFX JQ"
#define kPluginDescription "Apply AgX film emulation to RGB channels"
#define kPluginIdentifier "com.JQ.AgXEmulsion"
#define kPluginVersionMajor 1
#define kPluginVersionMinor 0

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

// Function to load JSON profile and handle nan values (same as test)
bool loadFilteredFilmLUT(const char* stock_name, float* logE, float* r, float* g, float* b) {
    std::string json_path = "/usr/OFX/Plugins/data/profiles/" + std::string(stock_name) + ".json";
    std::ifstream file(json_path);
    
    if (!file.is_open()) {
        printf("DEBUG: Could not open JSON profile: %s\n", json_path.c_str());
        return false;
    }
    
    try {
        // Read entire file into string first
        std::string json_content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();
        
        // Replace NaN with null for nlohmann/json compatibility
        size_t pos = 0;
        while ((pos = json_content.find("NaN", pos)) != std::string::npos) {
            json_content.replace(pos, 3, "null");
            pos += 4; // length of "null"
        }
        
        // Parse JSON using nlohmann/json library (matches Python json.load())
        json profile = json::parse(json_content);
        
        // Extract data arrays (matches Python implementation)
        auto& data = profile["data"];
        auto log_exposure_array = data["log_exposure"];
        auto density_curves_array = data["density_curves"];
        
        std::vector<float> temp_log_exposure;
        std::vector<float> temp_density_r, temp_density_g, temp_density_b;
        
        // Parse log_exposure array
        for (const auto& val : log_exposure_array) {
            temp_log_exposure.push_back(val.get<float>());
        }
        
        // Parse density_curves array (matches Python: list of [r,g,b] arrays)
        for (const auto& rgb_array : density_curves_array) {
            if (rgb_array.size() >= 3) {
                // Handle NaN values (same as Python implementation)
                float r_val = rgb_array[0].is_null() ? std::numeric_limits<float>::quiet_NaN() : rgb_array[0].get<float>();
                float g_val = rgb_array[1].is_null() ? std::numeric_limits<float>::quiet_NaN() : rgb_array[1].get<float>();
                float b_val = rgb_array[2].is_null() ? std::numeric_limits<float>::quiet_NaN() : rgb_array[2].get<float>();
                
                temp_density_r.push_back(r_val);
                temp_density_g.push_back(g_val);
                temp_density_b.push_back(b_val);
            }
        }
        
        // Handle nan values (same as Python implementation: replace with zeros)
        std::vector<float> log_exposure, density_r, density_g, density_b;
        for (size_t i = 0; i < temp_log_exposure.size(); i++) {
            if (i < temp_density_r.size()) {
                log_exposure.push_back(temp_log_exposure[i]);
                // Replace NaN with 0.0 (same as Python's np.nan_to_num)
                density_r.push_back(std::isnan(temp_density_r[i]) ? 0.0f : temp_density_r[i]);
                density_g.push_back(std::isnan(temp_density_g[i]) ? 0.0f : temp_density_g[i]);
                density_b.push_back(std::isnan(temp_density_b[i]) ? 0.0f : temp_density_b[i]);
            }
        }
        
        // Apply Python's nanmin subtraction (line 105 in emulsion.py)
        if (!density_r.empty()) {
            float r_min = 1e6, g_min = 1e6, b_min = 1e6;
            
            // Find minimum non-zero values (equivalent to nanmin)
            for (size_t i = 0; i < density_r.size(); i++) {
                if (density_r[i] > 0) r_min = std::min(r_min, density_r[i]);
                if (density_g[i] > 0) g_min = std::min(g_min, density_g[i]);
                if (density_b[i] > 0) b_min = std::min(b_min, density_b[i]);
            }
            
            // Subtract minimum (same as Python: self.density_curves -= np.nanmin(self.density_curves, axis=0))
            for (size_t i = 0; i < density_r.size(); i++) {
                density_r[i] = (density_r[i] > 0) ? (density_r[i] - r_min) : 0.0f;
                density_g[i] = (density_g[i] > 0) ? (density_g[i] - g_min) : 0.0f;
                density_b[i] = (density_b[i] > 0) ? (density_b[i] - b_min) : 0.0f;
            }
        }
        
        if (log_exposure.empty()) {
            printf("DEBUG: No valid data found in JSON profile after filtering\n");
            return false;
        }
        
        // Copy data to output arrays (limit to 601 samples)
        int count = std::min((int)log_exposure.size(), 601);
        for (int i = 0; i < count; i++) {
            logE[i] = log_exposure[i];
            r[i] = density_r[i];
            g[i] = density_g[i];
            b[i] = density_b[i];
        }
        
        // Pad with last value if needed
        for (int i = count; i < 601; i++) {
            logE[i] = logE[count-1];
            r[i] = r[count-1];
            g[i] = g[count-1];
            b[i] = b[count-1];
        }
        
        // Calculate actual min/max values (excluding zeros from NaN conversion)
        float r_min = 1e6, r_max = -1e6;
        float g_min = 1e6, g_max = -1e6;
        float b_min = 1e6, b_max = -1e6;
        
        for (int i = 0; i < count; i++) {
            if (r[i] > 0) {
                r_min = std::min(r_min, r[i]);
                r_max = std::max(r_max, r[i]);
            }
            if (g[i] > 0) {
                g_min = std::min(g_min, g[i]);
                g_max = std::max(g_max, g[i]);
            }
            if (b[i] > 0) {
                b_min = std::min(b_min, b[i]);
                b_max = std::max(b_max, b[i]);
            }
        }
        
        printf("DEBUG: Loaded JSON profile: %d valid samples (filtered from %zu total)\n", count, temp_log_exposure.size());
        printf("DEBUG: Log exposure range: [%f, %f]\n", logE[0], logE[count-1]);
        printf("DEBUG: Density R range: [%f, %f] (actual non-zero: [%f, %f])\n", r[0], r[count-1], r_min, r_max);
        printf("DEBUG: Density G range: [%f, %f] (actual non-zero: [%f, %f])\n", g[0], g[count-1], g_min, g_max);
        printf("DEBUG: Density B range: [%f, %f] (actual non-zero: [%f, %f])\n", b[0], b[count-1], b_min, b_max);

        // === Upload camera spectral sensitivity (81 samples) ===
        if(data.contains("log_sensitivity")){
            auto log_sens_array = data["log_sensitivity"];
            int tot = log_sens_array.size();
            std::vector<float> sensR81(81), sensG81(81), sensB81(81);
            for(int i=0;i<81;i++){
                int idx = (int)round(i*(tot-1)/80.0);
                auto arr = log_sens_array[idx];
                float lr = arr[0].is_null()? -10.0f : arr[0].get<float>();
                float lg = arr[1].is_null()? -10.0f : arr[1].get<float>();
                float lb = arr[2].is_null()? -10.0f : arr[2].get<float>();
                sensR81[i]=powf(10.0f, lr);
                sensG81[i]=powf(10.0f, lg);
                sensB81[i]=powf(10.0f, lb);
            }
            UploadCameraSensCUDA(sensR81.data(), sensG81.data(), sensB81.data(), 81);
        }
        return true;
        
    } catch (const std::exception& e) {
        printf("DEBUG: JSON parsing error: %s\n", e.what());
        return false;
    }
}

bool loadFilteredPaperLUT(const char* paper_name, float* logE, float* r, float* g, float* b) {
    std::string json_path = "/usr/OFX/Plugins/data/profiles/" + std::string(paper_name) + ".json";
    std::ifstream file(json_path);
    
    if (!file.is_open()) {
        printf("DEBUG: Could not open paper JSON profile: %s\n", json_path.c_str());
        return false;
    }
    
    try {
        // Read entire file into string first
        std::string json_content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();
        
        // Replace NaN with null for nlohmann/json compatibility
        size_t pos = 0;
        while ((pos = json_content.find("NaN", pos)) != std::string::npos) {
            json_content.replace(pos, 3, "null");
            pos += 4; // length of "null"
        }
        
        // Parse JSON using nlohmann/json library (matches Python json.load())
        json profile = json::parse(json_content);
        
        // Extract data arrays (matches Python implementation)
        auto& data = profile["data"];
        auto log_exposure_array = data["log_exposure"];
        auto density_curves_array = data["density_curves"];
        
        std::vector<float> temp_log_exposure;
        std::vector<float> temp_density_r, temp_density_g, temp_density_b;
        
        // Parse log_exposure array
        for (const auto& val : log_exposure_array) {
            temp_log_exposure.push_back(val.get<float>());
        }
        
        // Parse density_curves array (matches Python: list of [r,g,b] arrays)
        for (const auto& rgb_array : density_curves_array) {
            if (rgb_array.size() >= 3) {
                // Handle NaN values (same as Python implementation)
                float r_val = rgb_array[0].is_null() ? std::numeric_limits<float>::quiet_NaN() : rgb_array[0].get<float>();
                float g_val = rgb_array[1].is_null() ? std::numeric_limits<float>::quiet_NaN() : rgb_array[1].get<float>();
                float b_val = rgb_array[2].is_null() ? std::numeric_limits<float>::quiet_NaN() : rgb_array[2].get<float>();
                
                temp_density_r.push_back(r_val);
                temp_density_g.push_back(g_val);
                temp_density_b.push_back(b_val);
            }
        }
        
        // Handle nan values (same as Python implementation: replace with zeros)
        std::vector<float> log_exposure, density_r, density_g, density_b;
        for (size_t i = 0; i < temp_log_exposure.size(); i++) {
            if (i < temp_density_r.size()) {
                log_exposure.push_back(temp_log_exposure[i]);
                // Replace NaN with 0.0 (same as Python's np.nan_to_num)
                density_r.push_back(std::isnan(temp_density_r[i]) ? 0.0f : temp_density_r[i]);
                density_g.push_back(std::isnan(temp_density_g[i]) ? 0.0f : temp_density_g[i]);
                density_b.push_back(std::isnan(temp_density_b[i]) ? 0.0f : temp_density_b[i]);
            }
        }
        
        // Apply Python's nanmin subtraction (line 105 in emulsion.py)
        if (!density_r.empty()) {
            float r_min = 1e6, g_min = 1e6, b_min = 1e6;
            
            // Find minimum non-zero values (equivalent to nanmin)
            for (size_t i = 0; i < density_r.size(); i++) {
                if (density_r[i] > 0) r_min = std::min(r_min, density_r[i]);
                if (density_g[i] > 0) g_min = std::min(g_min, density_g[i]);
                if (density_b[i] > 0) b_min = std::min(b_min, density_b[i]);
            }
            
            // Subtract minimum (same as Python: self.density_curves -= np.nanmin(self.density_curves, axis=0))
            for (size_t i = 0; i < density_r.size(); i++) {
                density_r[i] = (density_r[i] > 0) ? (density_r[i] - r_min) : 0.0f;
                density_g[i] = (density_g[i] > 0) ? (density_g[i] - g_min) : 0.0f;
                density_b[i] = (density_b[i] > 0) ? (density_b[i] - b_min) : 0.0f;
            }
        }
        
        if (log_exposure.empty()) {
            printf("DEBUG: No valid data found in paper JSON profile after filtering\n");
            return false;
        }
        
        // Copy data to output arrays (limit to 601 samples)
        int count = std::min((int)log_exposure.size(), 601);
        for (int i = 0; i < count; i++) {
            logE[i] = log_exposure[i];
            r[i] = density_r[i];
            g[i] = density_g[i];
            b[i] = density_b[i];
        }
        
        // Pad with last value if needed
        for (int i = count; i < 601; i++) {
            logE[i] = logE[count-1];
            r[i] = r[count-1];
            g[i] = g[count-1];
            b[i] = b[count-1];
        }
        
        // Calculate actual min/max values (excluding zeros from NaN conversion)
        float r_min = 1e6, r_max = -1e6;
        float g_min = 1e6, g_max = -1e6;
        float b_min = 1e6, b_max = -1e6;
        
        for (int i = 0; i < count; i++) {
            if (r[i] > 0) {
                r_min = std::min(r_min, r[i]);
                r_max = std::max(r_max, r[i]);
            }
            if (g[i] > 0) {
                g_min = std::min(g_min, g[i]);
                g_max = std::max(g_max, g[i]);
            }
            if (b[i] > 0) {
                b_min = std::min(b_min, b[i]);
                b_max = std::max(b_max, b[i]);
            }
        }
        
        printf("DEBUG: Loaded paper JSON profile: %d valid samples (filtered from %zu total)\n", count, temp_log_exposure.size());
        printf("DEBUG: Paper log exposure range: [%f, %f]\n", logE[0], logE[count-1]);
        printf("DEBUG: Paper density R range: [%f, %f] (actual non-zero: [%f, %f])\n", r[0], r[count-1], r_min, r_max);
        printf("DEBUG: Paper density G range: [%f, %f] (actual non-zero: [%f, %f])\n", g[0], g[count-1], g_min, g_max);
        printf("DEBUG: Paper density B range: [%f, %f] (actual non-zero: [%f, %f])\n", b[0], b[count-1], b_min, b_max);

        // Upload dye spectra if available
        if(data.contains("dye_density")){
            auto dd_array = data["dye_density"];
            int tot = dd_array.size();
            std::vector<float> c81(81), m81(81), y81(81), dmin81(81);
            for(int i=0;i<81;i++){
                int idx = (int)round(i*(tot-1)/80.0);
                auto arr = dd_array[idx];
                float c= arr.size()>0 && !arr[0].is_null()? arr[0].get<float>() : 0.f;
                float m= arr.size()>1 && !arr[1].is_null()? arr[1].get<float>() : 0.f;
                float yv= arr.size()>2 && !arr[2].is_null()? arr[2].get<float>() : 0.f;
                float dmin = arr.size()>3 && !arr[3].is_null()? arr[3].get<float>() : 0.f;
                c81[i]=c; m81[i]=m; y81[i]=yv; dmin81[i]=dmin;
            }
            UploadPaperSpectraCUDA(c81.data(), m81.data(), y81.data(), dmin81.data(), 81);
        }
        return true;
        
    } catch (const std::exception& e) {
        printf("DEBUG: Paper JSON parsing error: %s\n", e.what());
        return false;
    }
}

bool loadSpectralLUTFromCSV(const char* csv_path, float* spectral_lut, int* width, int* height, int* spectral_samples) {
    std::ifstream file(csv_path);
    
    if (!file.is_open()) {
        printf("DEBUG: Could not open spectral LUT CSV: %s\n", csv_path);
        return false;
    }
    
    try {
        std::string line;
        std::vector<std::vector<float>> data;
        
        // Skip header line(s) - look for line that starts with a number
        while (std::getline(file, line)) {
            // Trim whitespace
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);
            
            // If line starts with a number, it's data (not header)
            if (!line.empty() && (isdigit(line[0]) || line[0] == '-')) {
                break;
            }
        }
        
        // Now parse the data line we found
        if (!line.empty()) {
            std::vector<float> row;
            std::istringstream iss(line);
            std::string value;
            
            // Parse CSV values
            while (std::getline(iss, value, ',')) {
                // Trim whitespace
                value.erase(0, value.find_first_not_of(" \t\r\n"));
                value.erase(value.find_last_not_of(" \t\r\n") + 1);
                
                if (!value.empty()) {
                    try {
                        row.push_back(std::stof(value));
                    } catch (const std::exception& e) {
                        printf("DEBUG: Failed to parse value '%s': %s\n", value.c_str(), e.what());
                        return false;
                    }
                }
            }
            
            if (!row.empty()) {
                data.push_back(row);
            }
        }
        
        // Continue reading remaining data lines
        while (std::getline(file, line)) {
            // Trim whitespace
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);
            
            if (line.empty()) continue;
            
            std::vector<float> row;
            std::istringstream iss(line);
            std::string value;
            
            // Parse CSV values
            while (std::getline(iss, value, ',')) {
                // Trim whitespace
                value.erase(0, value.find_first_not_of(" \t\r\n"));
                value.erase(value.find_last_not_of(" \t\r\n") + 1);
                
                if (!value.empty()) {
                    try {
                        row.push_back(std::stof(value));
                    } catch (const std::exception& e) {
                        printf("DEBUG: Failed to parse value '%s': %s\n", value.c_str(), e.what());
                        return false;
                    }
                }
            }
            
            if (!row.empty()) {
                data.push_back(row);
            }
        }
        
        if (data.empty()) {
            printf("DEBUG: No data found in spectral LUT CSV\n");
            return false;
        }
        
        printf("DEBUG: Parsed %zu data rows from CSV\n", data.size());
        printf("DEBUG: First row has %zu values\n", data[0].size());
        
        // Extract dimensions from metadata
        std::string meta_path = std::string(csv_path);
        meta_path = meta_path.substr(0, meta_path.find(".csv")) + "_meta.txt";
        
                                std::ifstream meta_file(meta_path);
                        if (meta_file.is_open()) {
                            std::string meta_line;
                            while (std::getline(meta_file, meta_line)) {
                                try {
                                    if (meta_line.find("original_shape_x=") == 0) {
                                        *width = std::stoi(meta_line.substr(16));
                                    } else if (meta_line.find("original_shape_y=") == 0) {
                                        *height = std::stoi(meta_line.substr(16));
                                    } else if (meta_line.find("spectral_samples=") == 0) {
                                        *spectral_samples = std::stoi(meta_line.substr(17));
                                    }
                                } catch (const std::exception& e) {
                                    printf("DEBUG: Failed to parse metadata line '%s': %s\n", meta_line.c_str(), e.what());
                                }
                            }
                            meta_file.close();
                        } else {
                            // Fallback: estimate from data
                            *width = 192;
                            *height = 192;
                            *spectral_samples = data[0].size() - 1; // -1 for x_y_index column
                        }
                        
                        // Ensure we have valid dimensions
                        if (*width <= 0 || *height <= 0) {
                            *width = 192;
                            *height = 192;
                        }
        
        // Copy data to output array (skip x_y_index column)
        int total_coordinates = data.size();
        int samples_per_coordinate = data[0].size() - 1;
        
        for (int i = 0; i < total_coordinates; i++) {
            for (int j = 0; j < samples_per_coordinate; j++) {
                spectral_lut[i * samples_per_coordinate + j] = data[i][j + 1]; // +1 to skip x_y_index
            }
        }
        
        printf("DEBUG: Loaded spectral LUT from CSV: %d coordinates, %d spectral samples\n", 
               total_coordinates, samples_per_coordinate);
        printf("DEBUG: Dimensions: %dx%d, spectral samples: %d\n", *width, *height, *spectral_samples);
        
        return true;
        
    } catch (const std::exception& e) {
        printf("DEBUG: Spectral LUT CSV parsing error: %s\n", e.what());
        return false;
    }
}

////////////////////////////////////////////////////////////////////////////////

class AgXEmulsionProcessor : public OFX::ImageProcessor
{
public:
    explicit AgXEmulsionProcessor(OFX::ImageEffect& p_Instance);

    virtual void processImagesCUDA();
    virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

    void setSrcImg(OFX::Image* p_SrcImg);
    void setImages(OFX::Image* src, OFX::Image* dst) {
        _srcImg = src; 
        _dstImg = dst;
    }
    // void setParams(float p_invertStrength);
    void setFrameTime(double frameTime) {_frameTime = frameTime; }
    void setFilmGamma(const char* film,float gamma){
        strncpy(_film, film, 63); 
        _film[63] = 0; 
        _gamma=gamma;
    }
    const char* getFilm() const {return _film;}
    float getGamma() const {return _gamma;}

    void setDiffusionHalation(float radius, float hal){ _radius = radius; _halStrength = hal; }
    void setGrain(float strength,unsigned int seed){ _grainStrength=strength; _grainSeed=seed; }
    void setExposure(float ev){ _exposureEV = ev; }
    void setPrintPaper(const char* paper){ strncpy(_paper,paper,63); _paper[63]=0; }
    void setDirParams(float amount,float interlayer,float diffUm,float highShift,float pxSize){ _dirAmount=amount; _dirInterlayer=interlayer; _dirDiffUm=diffUm; _dirHighShift=highShift; _pxSize=pxSize; }
    void setPrintParams(bool rm,float exp,float pf){ _removeGlareFlag=rm; _printExposure=exp; _preflash=pf; }

private:
    OFX::Image* _srcImg;
    // float _unused;
    double _frameTime;
    char _film[64];
    char _paper[64];
    float _gamma;
    float _radius;
    float _halStrength;
    float _grainStrength;
    unsigned int _grainSeed;
    float _exposureEV;
    // DIR parameters
    float _dirAmount{1.f};
    float _dirInterlayer{1.f};
    float _dirDiffUm{10.f};
    float _dirHighShift{0.f};
    float _pxSize{5.f};
    bool  _removeGlareFlag{true};
    float _printExposure{1.f};
    float _preflash{0.f};
};

AgXEmulsionProcessor::AgXEmulsionProcessor(OFX::ImageEffect& p_Instance)
    : OFX::ImageProcessor(p_Instance)
{
    _film[0] = '\0';  // Initialize as empty string
    _radius = 0.f;
    _halStrength = 0.f;
    _grainStrength = 0.f;
    _grainSeed = 0;
    _exposureEV = 0.f;
    _paper[0]='\0';
    _dirAmount=1.f; _dirInterlayer=1.f; _dirDiffUm=10.f; _dirHighShift=0.f; _pxSize=5.f;
}

extern "C" void LaunchEmulsionCUDA(float* img, int width, int height, float gamma, float exposureEV);
extern "C" void UploadExposureLUTCUDA(const uint16_t* lut,int sizeX,int sizeY);
extern "C" void UploadCameraMatrixCUDA(const float* m33);
extern "C" void LaunchDynamicSpectralUpsamplingCUDA(float* img,int width,int height);
extern "C" bool IsSpectralLUTValid();
extern "C" bool LoadSpectralLUTCUDA(const char* filename);
extern "C" void LaunchDirCouplerCUDA(float* img, int width, int height);
extern "C" void LaunchDiffusionHalationCUDA(float* img, int width, int height, float radius, float halStrength);
extern "C" void LaunchGrainCUDA(float* img, int width, int height, float strength, unsigned int seed);
extern "C" void LaunchPaperCUDA(float* img, int width, int height);
extern "C" void UploadLUTCUDA(const float* logE, const float* r, const float* g, const float* b);
extern "C" void UploadDirMatrixCUDA(const float* matrix);
extern "C" void UploadDirParamsCUDA(const float* dMax, float highShift, float sigmaPx);
extern "C" void UploadPaperLUTCUDA(const float* logE, const float* r, const float* g, const float* b);
extern "C" void UploadViewSPDCUDA(const float* spd, int count);
extern "C" void UploadCATScaleCUDA(const float* scale);
extern "C" void UploadPaperParamsCUDA(float exposure, float preflash);
// (forward declarations already at top)

void AgXEmulsionProcessor::processImagesCUDA()
{
    static char sFilm[64]="";
    static float sGamma=-1.0f;
    static time_t lastFilmModTime=0;
    bool lutOK=true;
    
    // Debug: Print current values
    printf("DEBUG: _film='%s', _gamma=%f\n", _film, _gamma);
    printf("DEBUG: sFilm='%s', sGamma=%f\n", sFilm, sGamma);
    
    // Check if we need to reload the film LUT
    bool needReload = false;
    if(_film[0]!='\0') {
        std::string json_path = "/usr/OFX/Plugins/data/profiles/" + std::string(_film) + ".json";
        struct stat st;
        if(stat(json_path.c_str(), &st) == 0) {
            if(strcmp(_film,sFilm)!=0 || _gamma!=sGamma || st.st_mtime > lastFilmModTime) {
                needReload = true;
                lastFilmModTime = st.st_mtime;
                printf("DEBUG: Film profile modified or parameters changed, need to reload\n");
            }
        }
    }
    
    if(needReload){
        printf("DEBUG: LUT needs update\n");
        
        // Load JSON profile with nan filtering (same as Python reference)
        float logE[601], r[601], g[601], b[601];
        bool loaded = loadFilteredFilmLUT(_film, logE, r, g, b);
        
        if (!loaded) {
            printf("ERROR: Failed to load JSON profile for %s\n", _film);
            printf("ERROR: No fallback available - JSON profile is required\n");
            lutOK = false;
        } else {
            printf("DEBUG: JSON profile loaded successfully\n");
            
            // Upload the film LUT to GPU
            UploadLUTCUDA(logE,r,g,b);
            
            // === DIR Coupler: compute and upload matrix & params ===
            // Compute density max per channel from film LUT
            float dMax[3]={0.f,0.f,0.f};
            for(int i=0;i<601;i++){dMax[0]=fmaxf(dMax[0],r[i]);dMax[1]=fmaxf(dMax[1],g[i]);dMax[2]=fmaxf(dMax[2],b[i]);}
            // Build DIR matrix using CPU helper (double precision)
            std::array<double,3> amtRGB = { (double)_dirAmount, (double)_dirAmount, (double)_dirAmount };
            cp::Matrix3 M = cp::compute_dir_couplers_matrix(amtRGB,(double)_dirInterlayer);
            float Mf[9];
            for(int row=0;row<3;row++) for(int col=0;col<3;col++) Mf[row*3+col] = (float)M[row][col];
            UploadDirMatrixCUDA(Mf);
            // Sigma in pixels
            float sigmaPx = (_pxSize>1e-6f)? (_dirDiffUm/_pxSize) : 0.f;
            UploadDirParamsCUDA(dMax,_dirHighShift,sigmaPx);
            strncpy(sFilm,_film,63); sFilm[63]=0;
            sGamma=_gamma;
            printf("DEBUG: LUT & DIR uploaded successfully (dMax=%f,%f,%f sigmaPx=%f)\n",dMax[0],dMax[1],dMax[2],sigmaPx);
        }
    } else {
        printf("DEBUG: Using cached LUT\n");
    }
    // Load print LUT if paper selected
    static char sPaper[64]="";
    bool paperOK = false;
    if(_paper[0]!='\0' && strcmp(_paper,sPaper)!=0){
        printf("DEBUG: Loading paper profile: %s\n", _paper);
        
        float printLogE[601],printR[601],printG[601],printB[601];
        bool paperLoaded = loadFilteredPaperLUT(_paper, printLogE, printR, printG, printB);
        
        if(paperLoaded) {
            if(_removeGlareFlag){
                auto applyGlare=[&](float* curve){
                    const int N=601; const float factor=0.2f; const float density=1.0f; const float transition=0.3f;
                    // compute mean curve
                    float mean[N]; for(int i=0;i<N;i++) mean[i]=(printR[i]+printG[i]+printB[i])*0.333333f;
                    // find le_center by interpolating mean curve at target density
                    // linear search
                    int idx=0; while(idx<N-1 && mean[idx]<density) idx++;
                    float le_center;
                    if(idx==0) le_center=printLogE[0];
                    else{
                        float t=(density-mean[idx-1])/(mean[idx]-mean[idx-1]+1e-6f);
                        le_center=printLogE[idx-1]*(1-t)+printLogE[idx]*t;
                    }
                    // slope measurement +-1 EV (log2 -> log10 factor)
                    float le_delta=0.30103f; // log10(2)/2 approx 0.1505? Wait in python they used log10(2**range_ev)/2 ; range=1 so delta=0.1505
                    le_delta=0.150515f;
                    float le0=le_center-le_delta, le1=le_center+le_delta;
                    auto interp=[&](float target)->float{
                        int j=0; while(j<N-1 && printLogE[j]<target) j++;
                        if(j==0) return curve[0]; if(j>=N-1) return curve[N-1];
                        float t=(target-printLogE[j-1])/(printLogE[j]-printLogE[j-1]+1e-6f);
                        return curve[j-1]*(1-t)+curve[j]*t; };
                    float density0=interp(le0);
                    float density1=interp(le1);
                    float slope=(density1-density0)/(le1-le0+1e-6f);
                    // create shifted le array
                    std::vector<float> le_nl(N);
                    for(int i=0;i<N;i++){
                        le_nl[i]=printLogE[i];
                        if(printLogE[i]>le_center) le_nl[i]-=(printLogE[i]-le_center)*factor;
                    }
                    // gaussian blur le_nl with sigma = (transition/slope)/le_step
                    float le_step=(printLogE[N-1]-printLogE[0])/(N-1);
                    float sigma=(transition/slope)/(le_step+1e-6f);
                    int rad=(int)(sigma*3); if(rad<1) rad=1; if(rad>50) rad=50;
                    std::vector<float> kernel(2*rad+1);
                    float sum=0.f; for(int k=-rad;k<=rad;k++){float w=expf(-0.5f*k*k/(sigma*sigma)); kernel[k+rad]=w; sum+=w;}
                    for(auto& w:kernel) w/=sum;
                    std::vector<float> tmp(N);
                    for(int i=0;i<N;i++){
                        float acc=0.f; for(int k=-rad;k<=rad;k++){int j=i+k; if(j<0) j=0; if(j>=N) j=N-1; acc+=le_nl[j]*kernel[k+rad];}
                        tmp[i]=acc; }
                    le_nl.swap(tmp);
                    // remap curve via interpolation
                    std::vector<float> outCurve(N);
                    for(int i=0;i<N;i++) outCurve[i]=interp(le_nl[i]);
                    for(int i=0;i<N;i++) curve[i]=outCurve[i];
                };
                applyGlare(printR);
                applyGlare(printG);
                applyGlare(printB);
            }
            
            // Upload paper curves
            UploadPaperLUTCUDA(printLogE,printR,printG,printB);
            printf("DEBUG: Paper LUT uploaded successfully\n");
            
            // Use D50 illuminant SPD (default for paper)
            UploadViewSPDCUDA(c_d50SPD, 81);
            printf("DEBUG: D50 illuminant SPD uploaded\n");
            
            // Compute CAT scale factors (D65 / D50)
            // Calculate D50 LMS from D50 SPD and Bradford matrix
            float X50=0.f,Y50=0.f,Z50=0.f,norm=0.f;
            for(int j=0;j<CIE_SAMPLES;j++){
                X50+=c_d50SPD[j]*c_xBar[j];
                Y50+=c_d50SPD[j]*c_yBar[j];
                Z50+=c_d50SPD[j]*c_zBar[j];
                norm+=c_d50SPD[j]*c_yBar[j];
            }
            if(norm>0){X50/=norm; Y50/=norm; Z50/=norm;}
            
            // Bradford transform to LMS
            float L50 = 0.8951f*X50 + 0.2664f*Y50 - 0.1614f*Z50;
            float M50 = -0.7502f*X50 + 1.7135f*Y50 + 0.0367f*Z50;
            float S50 = 0.0389f*X50 - 0.0685f*Y50 + 1.0296f*Z50;
            
            float scale[3];
            scale[0]=c_d65LMS[0]/L50; scale[1]=c_d65LMS[1]/M50; scale[2]=c_d65LMS[2]/S50;
            UploadCATScaleCUDA(scale);
            printf("DEBUG: CAT scale uploaded (%f,%f,%f)\n",scale[0],scale[1],scale[2]);
            
            // Upload paper parameters
            UploadPaperParamsCUDA(_printExposure,_preflash);
            printf("DEBUG: Paper parameters uploaded\n");
            
            paperOK = true;
            strncpy(sPaper,_paper,63); sPaper[63]=0;
        } else {
            printf("DEBUG: Failed to load paper profile: %s\n", _paper);
        }
    }

    static bool spectralLUTLoaded = false;
    static time_t lastModTime = 0;
    printf("DEBUG: spectralLUTLoaded = %s\n", spectralLUTLoaded ? "true" : "false");
    
    // Check if we need to reload the spectral LUT file
    struct stat st;
    const char* spectralLUTPath = "/usr/OFX/Plugins/AgXEmulsionPlugin.ofx.bundle/Contents/Linux-x86-64/data/irradiance_xy_tc.csv";
    bool spectralNeedReload = false;
    
    if(stat(spectralLUTPath, &st) == 0) {
        if(!spectralLUTLoaded || st.st_mtime > lastModTime) {
            spectralNeedReload = true;
            lastModTime = st.st_mtime;
            printf("DEBUG: Spectral LUT CSV file modified, need to reload\n");
        }
    } else {
        printf("DEBUG: Could not stat spectral LUT CSV file\n");
    }
    
    if(!spectralLUTLoaded || spectralNeedReload){
        printf("DEBUG: Attempting to load Spectral LUT from %s\n", spectralLUTPath);
        
        // Load spectral LUT from CSV
        int width, height, spectral_samples;
        float* spectral_lut = new float[36864 * 81]; // 192*192 * 81 spectral samples
        
        bool loaded = loadSpectralLUTFromCSV(spectralLUTPath, spectral_lut, &width, &height, &spectral_samples);
        
        if(loaded) {
            printf("DEBUG: Spectral LUT loaded from CSV successfully\n");
            UploadSpectralLUTCUDA(spectral_lut,width,height,spectral_samples);
            spectralLUTLoaded=true;
        } else {
            printf("ERROR: Failed to load spectral LUT from CSV, using fallback\n");
            // For now, we'll use hardcoded values that match Python reference
            spectralLUTLoaded = true;
        }
        
        delete[] spectral_lut;
    }

    const OfxRectI& bounds = _srcImg->getBounds();
    const int width  = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* srcPtr = static_cast<float*>(_srcImg->getPixelData());
    float* dstPtr = static_cast<float*>(_dstImg->getPixelData());
    size_t contigBytes = static_cast<size_t>(width) * height * 4 * sizeof(float);
    // Always use a device staging buffer for correctness
    float* devImg = nullptr;
    cudaError_t allocErr = cudaMalloc(&devImg, contigBytes);
    if(allocErr!=cudaSuccess){ printf("ERROR cudaMalloc devImg: %s\n", cudaGetErrorString(allocErr)); return; }
    cudaError_t cpyInErr = cudaMemcpy(devImg, srcPtr, contigBytes, cudaMemcpyHostToDevice);
    if(cpyInErr!=cudaSuccess){ printf("ERROR cudaMemcpy H2D: %s\n", cudaGetErrorString(cpyInErr)); cudaFree(devImg); return; }

    if(spectralLUTLoaded){
        bool textureValid = IsSpectralLUTValid();
        printf("DEBUG: Spectral LUT texture valid = %s\n", textureValid ? "true" : "false");
        if(textureValid) {
            LaunchDynamicSpectralUpsamplingCUDA(devImg,width,height);
            // Debug: sample center after Spectral LUT
            {
                size_t center = (size_t)(height/2) * width + width/2;
                float sample[4];
                cudaMemcpy(sample, devImg + center*4, 3*sizeof(float), cudaMemcpyDeviceToHost);
                printf("DEBUG: After Spectral LUT center(CMY) = %f,%f,%f\n", sample[0], sample[1], sample[2]);
            }
        } else {
            printf("ERROR: Spectral LUT texture not uploaded! Skipping Spectral LUT stage.\n");
        }
    } else {
        printf("WARNING: Spectral LUT texture not uploaded! Skipping Spectral LUT stage.\n");
    }

    if(lutOK){
        // === Negative development with DIR couplers ===
        // Step 1: logE ➜ CMY density
        printf("DEBUG: Launching Emulsion kernel with gamma=%f, exposureEV=%f\n", _gamma, _exposureEV);
        LaunchEmulsionCUDA(devImg, width, height, _gamma, _exposureEV);
        // Debug: sample center after Emulsion (logE/gamma)
        {
            size_t center = (size_t)(height/2) * width + width/2;
            float sample[4];
            cudaMemcpy(sample, devImg + center*4, 3*sizeof(float), cudaMemcpyDeviceToHost);
            printf("DEBUG: After Emulsion center(logE/gamma) = %f,%f,%f\n", sample[0], sample[1], sample[2]);
        }
        // Step 2: apply DIR inhibition, converts back to light
        printf("DEBUG: Launching DIR Coupler kernel\n");
        LaunchDirCouplerCUDA(devImg,width,height);
        // Debug: sample center after DIR (linear RGB)
        {
            size_t center = (size_t)(height/2) * width + width/2;
            float sample[4];
            cudaMemcpy(sample, devImg + center*4, 3*sizeof(float), cudaMemcpyDeviceToHost);
            printf("DEBUG: After DIR center(RGB) = %f,%f,%f\n", sample[0], sample[1], sample[2]);
        }
        // Apply diffusion + halation if enabled
        if(_radius > 0.1f || _halStrength > 1e-5f){
            printf("DEBUG: LaunchDiffusionHalation radius=%f hal=%f\n", _radius, _halStrength);
            LaunchDiffusionHalationCUDA(devImg, width, height, _radius, _halStrength);
            // Debug: sample center after Diffusion/Halation
            {
                size_t center = (size_t)(height/2) * width + width/2;
                float sample[4];
                cudaMemcpy(sample, devImg + center*4, 3*sizeof(float), cudaMemcpyDeviceToHost);
                printf("DEBUG: After Diffusion/Halation center(RGB) = %f,%f,%f\n", sample[0], sample[1], sample[2]);
            }
        }

        if(_grainStrength > 1e-5f){
            printf("DEBUG: LaunchGrain strength=%f seed=%u\n", _grainStrength, _grainSeed);
            LaunchGrainCUDA(devImg, width, height, _grainStrength, _grainSeed);
            // Debug: sample center after Grain
            {
                size_t center = (size_t)(height/2) * width + width/2;
                float sample[4];
                cudaMemcpy(sample, devImg + center*4, 3*sizeof(float), cudaMemcpyDeviceToHost);
                printf("DEBUG: After Grain center(RGB) = %f,%f,%f\n", sample[0], sample[1], sample[2]);
            }
        }

        if(_paper[0]!='\0' && paperOK){
            printf("DEBUG: Launching Paper kernel with paper=%s\n", _paper);
            LaunchPaperCUDA(devImg,width,height);
            // Debug: sample center after Paper
            {
                size_t center = (size_t)(height/2) * width + width/2;
                float sample[4];
                cudaMemcpy(sample, devImg + center*4, 3*sizeof(float), cudaMemcpyDeviceToHost);
                printf("DEBUG: After Paper center(sRGB) = %f,%f,%f\n", sample[0], sample[1], sample[2]);
            }
        } else {
            printf("WARNING: Paper not selected or not loaded! Skipping Paper stage.\n");
        }
    } else {
        printf("ERROR: Film LUT not loaded! Skipping all film processing stages.\n");
    }

    // Copy result back to host
    cudaError_t cpyOutErr = cudaMemcpy(dstPtr, devImg, contigBytes, cudaMemcpyDeviceToHost);
    if(cpyOutErr!=cudaSuccess){ printf("ERROR cudaMemcpy D2H: %s\n", cudaGetErrorString(cpyOutErr)); }
    cudaFree(devImg);
}

void AgXEmulsionProcessor::multiThreadProcessImages(OfxRectI p_ProcWindow)
{
    for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y)
    {
        if (_effect.abort()) break;

        float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));

        for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x)
        {
            float* srcPix = static_cast<float*>(_srcImg ? _srcImg->getPixelAddress(x, y) : 0);

            // do we have a source image to scale up
            if (srcPix)
            {
                dstPix[0] = srcPix[0];
                dstPix[1] = srcPix[1];
                dstPix[2] = srcPix[2];
                dstPix[3] = srcPix[3];
            }
            else
            {
                // no src pixel here, be black and transparent
                for (int c = 0; c < 4; ++c)
                {
                    dstPix[c] = 0;
                }
            }

            // increment the dst pixel
            dstPix += 4;
        }
    }
}

void AgXEmulsionProcessor::setSrcImg(OFX::Image* p_SrcImg)
{
    _srcImg = p_SrcImg;
}


////////////////////////////////////////////////////////////////////////////////
/** @brief The plugin that does our work */
class AgXEmulsionPlugin : public OFX::ImageEffect
{
public:
    explicit AgXEmulsionPlugin(OfxImageEffectHandle p_Handle);

    /* Override the render */
    virtual void render(const OFX::RenderArguments& p_Args);

    /* Override is identity */
    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);

    /* Set up and run a processor */
    void setupAndProcess(AgXEmulsionProcessor &p_AgXProcessor, const OFX::RenderArguments& p_Args);

private:
    // Does not own the following pointers
    OFX::Clip* m_DstClip;
    OFX::Clip* m_SrcClip;

    OFX::ChoiceParam* m_filmStock;
    OFX::DoubleParam* m_gammaFactor;
    OFX::DoubleParam* m_exposureEV;
    OFX::ChoiceParam* m_printPaper;
    OFX::ChoiceParam* m_inputColorSpace;
    OFX::BooleanParam* m_removeGlare;
    OFX::DoubleParam* m_printExposure;
    OFX::DoubleParam* m_preflash;
    OFX::DoubleParam* m_diffusionRadius;
    OFX::DoubleParam* m_halationStrength;
    OFX::DoubleParam* m_grainStrength;
    OFX::IntParam*    m_grainSeed;
    OFX::DoubleParam* m_dirAmount;
    OFX::DoubleParam* m_dirInterlayer;
    OFX::DoubleParam* m_dirDiffUm;
    OFX::DoubleParam* m_dirHighShift;
    OFX::DoubleParam* m_pixelSizeUm;
};

AgXEmulsionPlugin::AgXEmulsionPlugin(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
{
    m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
    m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

    m_filmStock = fetchChoiceParam("filmStock");
    m_gammaFactor = fetchDoubleParam("gammaFactor");
    m_exposureEV = fetchDoubleParam("exposureEV");
    m_printPaper = fetchChoiceParam("printPaper");
    m_inputColorSpace = fetchChoiceParam("inputColorSpace");
    m_removeGlare = fetchBooleanParam("removeGlare");
    m_printExposure = fetchDoubleParam("printExposure");
    m_preflash = fetchDoubleParam("preflashExposure");
    m_diffusionRadius = fetchDoubleParam("diffusionRadius");
    m_halationStrength = fetchDoubleParam("halationStrength");
    m_grainStrength   = fetchDoubleParam("grainStrength");
    m_grainSeed       = fetchIntParam("grainSeed");
    m_dirAmount = fetchDoubleParam("dirAmount");
    m_dirInterlayer = fetchDoubleParam("dirInterlayer");
    m_dirDiffUm = fetchDoubleParam("dirDiffusionUm");
    m_dirHighShift = fetchDoubleParam("dirHighShift");
    m_pixelSizeUm = fetchDoubleParam("pixelSizeUm");
}

void AgXEmulsionPlugin::render(const OFX::RenderArguments& p_Args)
{
    if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
    {
        AgXEmulsionProcessor agxProcessor(*this);
        agxProcessor.setFrameTime(p_Args.time);
        setupAndProcess(agxProcessor, p_Args);
    }
    else
    {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
}

bool AgXEmulsionPlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
    if (m_SrcClip)
    {
        p_IdentityClip = m_SrcClip;
        p_IdentityTime = p_Args.time;
        return true;
    }
    return false;
}

void AgXEmulsionPlugin::setupAndProcess(AgXEmulsionProcessor& p_AgXProcessor, const OFX::RenderArguments& p_Args)
{
    // Get the render window and the time from the render arguments
    const OfxTime time = p_Args.time;
    const OfxRectI& renderWindow = p_Args.renderWindow;

    // Retrieve any instance data associated with this effect
    OFX::Image* src = m_SrcClip->fetchImage(time);
    OFX::Image* dst = m_DstClip->fetchImage(time);

    // Set images and other arguments
    p_AgXProcessor.setImages(src, dst);

    // Get parameter values
    int filmIdx; 
    m_filmStock->getValueAtTime(p_Args.time, filmIdx);
    printf("DEBUG: Film index from param = %d\n", filmIdx);
    
    const char* filmName = filmIdx==0?"kodak_portra_400":"kodak_vision3_250d";
    printf("DEBUG: Film name = %s\n", filmName);

    int csIdx;
    m_inputColorSpace->getValueAtTime(p_Args.time, csIdx);
    const char* csName = csIdx==0?"ACES2065-1":"Unknown";
    printf("DEBUG: Input color space = %s\n", csName);

    bool removeGlare;
    m_removeGlare->getValueAtTime(p_Args.time, removeGlare);
    printf("DEBUG: removeGlare=%d\n", removeGlare);

    double printExposure = m_printExposure->getValueAtTime(p_Args.time);
    double preflash = m_preflash->getValueAtTime(p_Args.time);
    printf("DEBUG: printExposure=%f preflash=%f\n",printExposure,preflash);
    
    double gamma = m_gammaFactor->getValueAtTime(p_Args.time);
    double exposure = m_exposureEV->getValueAtTime(p_Args.time);
    double radius = m_diffusionRadius->getValueAtTime(p_Args.time);
    double hal   = m_halationStrength->getValueAtTime(p_Args.time);
    double grain = m_grainStrength->getValueAtTime(p_Args.time);
    int    gseed = m_grainSeed->getValueAtTime(p_Args.time);
    int paperIdx; m_printPaper->getValueAtTime(p_Args.time,paperIdx);
    const char* paperMap[] = {"", "kodak_2383", "kodak_2393", "fujifilm_crystal_archive_typeii", "kodak_ektacolor_edge", "kodak_endura_premier", "kodak_portra_endura", "kodak_supra_endura", "kodak_ultra_endura"};
    const char* paperName = paperIdx<9?paperMap[paperIdx]:"";
    printf("DEBUG: PrintPaper idx=%d name=%s\n",paperIdx,paperName);
    printf("DEBUG: ExposureEV = %f Radius = %f Halation = %f\n", exposure, radius, hal);
    printf("DEBUG: Grain strength = %f seed=%d\n", grain, gseed);
 
    p_AgXProcessor.setFilmGamma(filmName,(float)gamma);
    p_AgXProcessor.setDiffusionHalation((float)radius,(float)hal);
    p_AgXProcessor.setExposure((float)exposure);
    p_AgXProcessor.setPrintPaper(paperName);
    p_AgXProcessor.setGrain((float)grain,(unsigned int)gseed);

    double dirAmt    = m_dirAmount->getValueAtTime(p_Args.time);
    double interLay  = m_dirInterlayer->getValueAtTime(p_Args.time);
    double diffUm    = m_dirDiffUm->getValueAtTime(p_Args.time);
    double highShift = m_dirHighShift->getValueAtTime(p_Args.time);
    double pxSize    = m_pixelSizeUm->getValueAtTime(p_Args.time);
    printf("DEBUG: DIR params amt=%f inter=%f diffUm=%f highShift=%f pxSize=%f\n",dirAmt,interLay,diffUm,highShift,pxSize);
    p_AgXProcessor.setDirParams((float)dirAmt,(float)interLay,(float)diffUm,(float)highShift,(float)pxSize);

    // after computing removeGlare etc in setupAndProcess before calling processor we call
     p_AgXProcessor.setPrintParams(removeGlare,(float)printExposure,(float)preflash);

    // Setup OpenCL and CUDA Render arguments
    p_AgXProcessor.setGPURenderArgs(p_Args);

    // Set the render window
    p_AgXProcessor.setRenderWindow(p_Args.renderWindow);

    // Call the base class process member, this will call the derived templated process code
    p_AgXProcessor.process();
}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

AgXEmulsionPluginFactory::AgXEmulsionPluginFactory()
    : OFX::PluginFactoryHelper<AgXEmulsionPluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void AgXEmulsionPluginFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
{
    // Basic labels
    p_Desc.setLabels(kPluginName, kPluginName, kPluginName);
    p_Desc.setPluginGrouping(kPluginGrouping);
    p_Desc.setPluginDescription(kPluginDescription);

    // Add the supported contexts, only filter at the moment
    p_Desc.addSupportedContext(eContextFilter);
    p_Desc.addSupportedContext(eContextGeneral);

    // Add supported pixel depths
    p_Desc.addSupportedBitDepth(eBitDepthFloat);

    // Set a few flags
    p_Desc.setSingleInstance(false);
    p_Desc.setHostFrameThreading(false);
    p_Desc.setSupportsMultiResolution(kSupportsMultiResolution);
    p_Desc.setSupportsTiles(kSupportsTiles);
    p_Desc.setTemporalClipAccess(false);
    p_Desc.setRenderTwiceAlways(false);
    p_Desc.setSupportsMultipleClipPARs(kSupportsMultipleClipPARs);

    // Setup CUDA render capability flags on non-Apple system
#ifndef __APPLE__
    p_Desc.setSupportsCudaRender(true);
    p_Desc.setSupportsCudaStream(false);
#endif
}

void AgXEmulsionPluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
{
    // Source clip only in the filter context
    // Create the mandated source clip
    ClipDescriptor* srcClip = p_Desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    srcClip->addSupportedComponent(ePixelComponentRGBA);
    srcClip->setTemporalClipAccess(false);
    srcClip->setSupportsTiles(kSupportsTiles);
    srcClip->setIsMask(false);

    // Create the mandated output clip
    ClipDescriptor* dstClip = p_Desc.defineClip(kOfxImageEffectOutputClipName);
    dstClip->addSupportedComponent(ePixelComponentRGBA);
    dstClip->addSupportedComponent(ePixelComponentAlpha);
    dstClip->setSupportsTiles(kSupportsTiles);

    // Make some pages and to things in
    PageParamDescriptor* page = p_Desc.definePageParam("Controls");

    // Input color space choice
    ChoiceParamDescriptor* cschoice = p_Desc.defineChoiceParam("inputColorSpace");
    cschoice->setLabel("Input Color Space");
    cschoice->setHint("Linear RGB colour space of source clip");
    cschoice->appendOption("Linear ACES2065-1 (AP0)");
    cschoice->setDefault(0);
    page->addChild(*cschoice);

    // Remove viewing glare toggle
    BooleanParamDescriptor* glareToggle = p_Desc.defineBooleanParam("removeGlare");
    glareToggle->setLabel("Remove Viewing Glare");
    glareToggle->setHint("Apply viewing glare compensation removal to print density curves");
    glareToggle->setDefault(true);
    page->addChild(*glareToggle);

    // Film stock choice
    ChoiceParamDescriptor* fchoice = p_Desc.defineChoiceParam("filmStock");
    fchoice->setLabel("Film Stock");
    fchoice->appendOption("Kodak Portra 400");
    fchoice->appendOption("Kodak Vision3 250D");
    fchoice->setDefault(0);
    page->addChild(*fchoice);

    // Exposure EV
    DoubleParamDescriptor* eparam = p_Desc.defineDoubleParam("exposureEV");
    eparam->setLabel("Exposure EV");
    eparam->setHint("Exposure adjustment in stops (log2)");
    eparam->setDefault(0.0);
    eparam->setRange(-4.0,4.0);
    eparam->setIncrement(0.1);
    eparam->setDisplayRange(-4.0,4.0);
    page->addChild(*eparam);

    // Gamma factor
    DoubleParamDescriptor* gparam = p_Desc.defineDoubleParam("gammaFactor");
    gparam->setLabel("Gamma Factor");
    gparam->setHint("Density curve gamma scaling");
    gparam->setDefault(1.0);
    gparam->setRange(0.5,2.0);
    gparam->setIncrement(0.01);
    gparam->setDisplayRange(0.5,2.0);
    page->addChild(*gparam);

    // Diffusion radius
    DoubleParamDescriptor* dparam = p_Desc.defineDoubleParam("diffusionRadius");
    dparam->setLabel("Diffusion Radius");
    dparam->setHint("Gaussian diffusion radius in pixels");
    dparam->setDefault(3.0);
    dparam->setRange(0.0,25.0);
    dparam->setIncrement(0.1);
    dparam->setDisplayRange(0.0,25.0);
    page->addChild(*dparam);

    // Halation strength
    DoubleParamDescriptor* hparam = p_Desc.defineDoubleParam("halationStrength");
    hparam->setLabel("Halation Strength");
    hparam->setHint("Amount of red halation to add");
    hparam->setDefault(0.2);
    hparam->setRange(0.0,1.0);
    hparam->setIncrement(0.01);
    hparam->setDisplayRange(0.0,1.0);
    page->addChild(*hparam);

    // Grain strength
    DoubleParamDescriptor* gs = p_Desc.defineDoubleParam("grainStrength");
    gs->setLabel("Grain Strength");
    gs->setHint("Amount of grain noise");
    gs->setDefault(0.1);
    gs->setRange(0.0,1.0);
    gs->setIncrement(0.01);
    gs->setDisplayRange(0.0,1.0);
    page->addChild(*gs);

    // Grain seed
    IntParamDescriptor* gseed = p_Desc.defineIntParam("grainSeed");
    gseed->setLabel("Grain Seed");
    gseed->setHint("Random seed for grain pattern");
    gseed->setDefault(0);
    gseed->setRange(0,100000);
    gseed->setDisplayRange(0,100000);
    page->addChild(*gseed);

    // Print paper choice
    ChoiceParamDescriptor* pchoice = p_Desc.defineChoiceParam("printPaper");
    pchoice->setLabel("Print Paper");
    pchoice->appendOption("None");
    pchoice->appendOption("Kodak 2383");
    pchoice->appendOption("Kodak 2393");
    pchoice->appendOption("Fujifilm Crystal Archive Type II");
    pchoice->appendOption("Kodak Ektacolor Edge");
    pchoice->appendOption("Kodak Endura Premier");
    pchoice->appendOption("Kodak Portra Endura");
    pchoice->appendOption("Kodak Supra Endura");
    pchoice->appendOption("Kodak Ultra Endura");
    pchoice->setDefault(1);
    page->addChild(*pchoice);

    // DIR couplers controls
    DoubleParamDescriptor* dirAmt = p_Desc.defineDoubleParam("dirAmount");
    dirAmt->setLabel("DIR Amount");
    dirAmt->setHint("Global DIR coupler strength");
    dirAmt->setDefault(1.0);
    dirAmt->setRange(0.0,2.0);
    dirAmt->setIncrement(0.01);
    dirAmt->setDisplayRange(0.0,2.0);
    page->addChild(*dirAmt);

    DoubleParamDescriptor* dirInter = p_Desc.defineDoubleParam("dirInterlayer");
    dirInter->setLabel("DIR σ Interlayer");
    dirInter->setHint("Inter-layer diffusion sigma (layers)");
    dirInter->setDefault(1.0);
    dirInter->setRange(0.0,3.0);
    dirInter->setIncrement(0.1);
    dirInter->setDisplayRange(0.0,3.0);
    page->addChild(*dirInter);

    DoubleParamDescriptor* dirDiff = p_Desc.defineDoubleParam("dirDiffusionUm");
    dirDiff->setLabel("DIR Diffusion μm");
    dirDiff->setHint("XY diffusion blur sigma in micrometers");
    dirDiff->setDefault(10.0);
    dirDiff->setRange(0.0,50.0);
    dirDiff->setIncrement(0.5);
    dirDiff->setDisplayRange(0.0,50.0);
    page->addChild(*dirDiff);

    DoubleParamDescriptor* dirShift = p_Desc.defineDoubleParam("dirHighShift");
    dirShift->setLabel("DIR High-Exposure Shift");
    dirShift->setHint("Non-linear saturation shift for high exposures");
    dirShift->setDefault(0.0);
    dirShift->setRange(0.0,1.0);
    dirShift->setIncrement(0.01);
    dirShift->setDisplayRange(0.0,1.0);
    page->addChild(*dirShift);

    DoubleParamDescriptor* pxParam = p_Desc.defineDoubleParam("pixelSizeUm");
    pxParam->setLabel("Pixel Size μm");
    pxParam->setHint("Pixel pitch in micrometers (sensor scanned)");
    pxParam->setDefault(5.0);
    pxParam->setRange(0.1,100.0);
    pxParam->setIncrement(0.1);
    pxParam->setDisplayRange(0.1,100.0);
    page->addChild(*pxParam);

    // Print exposure slider
    DoubleParamDescriptor* pexp = p_Desc.defineDoubleParam("printExposure");
    pexp->setLabel("Print Exposure");
    pexp->setHint("Relative exposure multiplier for print projection");
    pexp->setDefault(1.0);
    pexp->setRange(0.1,4.0);
    pexp->setIncrement(0.01);
    pexp->setDisplayRange(0.1,4.0);
    page->addChild(*pexp);

    DoubleParamDescriptor* pflash = p_Desc.defineDoubleParam("preflashExposure");
    pflash->setLabel("Preflash Exposure");
    pflash->setHint("Additive pre-flash light level (0-0.1)");
    pflash->setDefault(0.0);
    pflash->setRange(0.0,0.1);
    pflash->setIncrement(0.001);
    pflash->setDisplayRange(0.0,0.1);
    page->addChild(*pflash);
}

ImageEffect* AgXEmulsionPluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
    return new AgXEmulsionPlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    static AgXEmulsionPluginFactory agxEmulsionPlugin;
    p_FactoryArray.push_back(&agxEmulsionPlugin);
} 