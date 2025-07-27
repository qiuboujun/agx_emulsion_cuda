# ---------------------- paths ----------------------
<<<<<<< HEAD
CUDA_PATH  ?= /slm/home/jqiu/Library/cuda
EIGEN_PATH ?= /usr/local/include/eigen3
=======
CUDA_PATH  ?= /usr/local/cuda
EIGEN_PATH ?= /usr/include/eigen3
>>>>>>> fbf8680 (initial commit for Cursor agent contextcd)
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
SOURCES_CU := gp_numpy.cu gp_scipy.cu $(UTILS_CU)
SOURCES_CPP:= main.cpp

OBJECTS_CU  := $(SOURCES_CU:.cu=.o)
OBJECTS_CPP := $(SOURCES_CPP:.cpp=.o)
OBJECTS     := $(OBJECTS_CU) $(OBJECTS_CPP)
DEPS        := $(OBJECTS:.o=.d)

# ---------------------- rules ----------------------
.PHONY: all clean tests
all: $(TARGET)

TEST_EXE := tests/norm_test

tests: tests/gen_norm_reference.py $(TEST_EXE)
	python3 tests/gen_norm_reference.py
	./$(TEST_EXE)

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
	rm -f $(TARGET) $(OBJECTS) $(DEPS) $(TEST_EXE)

