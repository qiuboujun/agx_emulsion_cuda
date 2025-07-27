# ---------------------- paths ----------------------
CUDA_PATH  ?= /usr/local/cuda
EIGEN_PATH ?= /usr/include/eigen3
UTILS_DIR  := utils

CXX  := g++
NVCC := $(CUDA_PATH)/bin/nvcc

# ---------------------- flags ----------------------
COMMON_DEFS := -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
INC_DIRS    := -I. -I$(EIGEN_PATH) -I$(CUDA_PATH)/include -I$(UTILS_DIR)

CXXFLAGS  := -std=c++17 -O2 -fPIC $(INC_DIRS) $(COMMON_DEFS) -MMD -MP
NVCCFLAGS := -std=c++17 -O2 $(INC_DIRS) $(COMMON_DEFS)                  \
             --expt-extended-lambda -use_fast_math                      \
             -Xcompiler -fPIC -MMD -MP

LDFLAGS := -L$(CUDA_PATH)/lib64 -Wl,-rpath,$(CUDA_PATH)/lib64           \
           -lcudart -lcufft

# ---------------------- sources --------------------
TARGET := gp_numpy_app

UTILS_CU   := $(wildcard $(UTILS_DIR)/*.cu)
# add gaussian filter CUDA source
SOURCES_CU := gp_numpy.cu gp_scipy.cu gaussian_filter_kernel.cu $(UTILS_CU)
# main application sources
SOURCES_CPP:= main.cpp color_models.cpp

OBJECTS_CU  := $(SOURCES_CU:.cu=.o)
OBJECTS_CPP := $(SOURCES_CPP:.cpp=.o)
OBJECTS     := $(OBJECTS_CU) $(OBJECTS_CPP)
LIB_OBJECTS := $(filter-out main.o,$(OBJECTS_CPP))
DEPS        := $(OBJECTS:.o=.d)

# ---------------------- rules ----------------------
.PHONY: all clean tests
all: $(TARGET)

TEST_EXES := tests/norm_test tests/gaussian_test tests/interp_test tests/spline_test tests/color_test

tests: tests/gen_norm_reference.py tests/gen_gaussian_reference.py tests/gen_interp_reference.py tests/gen_color_reference.py $(TEST_EXES)
	python3 tests/gen_norm_reference.py
	python3 tests/gen_gaussian_reference.py
	python3 tests/gen_interp_reference.py
	python3 tests/gen_color_reference.py
	@echo "Running C++/CUDA tests..."
	@for t in $(TEST_EXES); do \
		./$$t || exit 1; \
	done

$(TARGET): $(OBJECTS)
	$(CXX) $^ -o $@ $(LDFLAGS)

# %.cu -> %.o
%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# %.cpp -> %.o
%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# include dependency files if they exist
-include $(DEPS)

clean:
	rm -f $(TARGET) $(OBJECTS) $(DEPS) $(TEST_EXES)

# ------------ build test executables --------------
# Norm PDF/CDF test
tests/norm_test: tests/test_norm_pdf_cdf.cpp $(OBJECTS_CU)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Gaussian filter test
tests/gaussian_test: tests/test_gaussian_filter.cpp $(OBJECTS_CU)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Interp1d test
tests/interp_test: tests/test_interp1d.cpp $(OBJECTS_CU)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Cubic spline test
tests/spline_test: tests/test_spline.cpp $(OBJECTS_CU)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# color model test
tests/color_test: tests/test_color_models.cpp $(OBJECTS_CU) $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

