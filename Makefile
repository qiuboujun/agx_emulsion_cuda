# Makefile for compiling the multi-file CUDA Gaussian Filter example

# --- Variables ---
# Use the CUDA_PATH from the user's environment if it exists, otherwise default.
CUDA_PATH ?= /slm/home/jqiu/Library/cuda
EIGEN_PATH ?= /usr/local/include/eigen3

# --- Tools ---
CXX = g++
NVCC = $(CUDA_PATH)/bin/nvcc

# --- Flags ---
# C++ compiler flags for .cpp files
CXXFLAGS = -std=c++17 -fPIC -I$(EIGEN_PATH) -I$(CUDA_PATH)/include

# NVCC compiler flags for .cu files
# -D EIGEN_DONT_USE_CUDA is important to prevent Eigen/nvcc header conflicts.
NVCCFLAGS = -std=c++17 --compiler-options="-fPIC" -use_fast_math -D EIGEN_DONT_USE_CUDA

# Linker flags - passed during the final linking stage.
# -Wl,-rpath,... embeds the library path into the executable.
LDFLAGS = -L$(CUDA_PATH)/lib64 -Wl,-rpath,$(CUDA_PATH)/lib64 -lcudart

# --- Targets ---
TARGET = gaussian_filter_app
SOURCES_CPP = main.cpp
SOURCES_CU = gaussian_filter_kernel.cu

# Automatically generate object file names from source file names
OBJECTS_CPP = $(SOURCES_CPP:.cpp=.o)
OBJECTS_CU = $(SOURCES_CU:.cu=.o)
OBJECTS = $(OBJECTS_CPP) $(OBJECTS_CU)

# --- Build Rules ---

# The default rule, executed when you just type 'make'.
all: $(TARGET)

# Rule to link the final executable from all the object files.
$(TARGET): $(OBJECTS)
	@echo "--- Linking executable: $@ ---"
	$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "--- Build complete. Executable is: $(TARGET) ---"

# Rule to compile .cpp files into .o object files.
%.o: %.cpp kernels.h
	@echo "--- Compiling C++ code: $< ---"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to compile .cu files into .o object files.
%.o: %.cu kernels.h
	@echo "--- Compiling CUDA code: $< ---"
	$(NVCC) $(NVCCFLAGS) -I$(EIGEN_PATH) -c $< -o $@

# Rule to clean up the build files.
clean:
	@echo "--- Cleaning up build files ---"
	rm -f $(TARGET) $(OBJECTS)

# .PHONY declares targets that are not actual files.
.PHONY: all clean

