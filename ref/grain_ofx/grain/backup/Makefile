UNAME_SYSTEM := $(shell uname -s)
CXX = g++
CXXFLAGS = -I../OpenFX-1.4/include -I../Support/include -Wall -Wextra

ifeq ($(UNAME_SYSTEM), Linux)
	AMDAPP_PATH ?= /opt/AMDAPP
	CXXFLAGS += -I${AMDAPP_PATH}/include -fPIC
	CUDAPATH ?= /home/resolve/cuda
	NVCC = $(CUDAPATH)/bin/nvcc
	NVCCFLAGS = --compiler-options="-fPIC" -use_fast_math
	LDFLAGS = -shared -L${CUDAPATH}/lib64 -lcuda -lcudart
	BUNDLE_DIR = GrainPlugin.ofx.bundle/Contents/Linux-x86-64/
	CUDA_OBJ = CudaKernel.o
endif

GrainPlugin.ofx: GrainPlugin.o ${CUDA_OBJ} ofxsCore.o ofxsImageEffect.o ofxsInteract.o ofxsLog.o ofxsMultiThread.o ofxsParams.o ofxsProperty.o ofxsPropertyValidation.o
	$(CXX) $^ -o $@ $(LDFLAGS)
	mkdir -p $(BUNDLE_DIR)
	cp GrainPlugin.ofx $(BUNDLE_DIR)

CudaKernel.o: CudaKernel.cu
	${NVCC} -c $< $(NVCCFLAGS)

%.o: ../Support/Library/%.cpp
	$(CXX) -c $< $(CXXFLAGS)

clean:
	rm -f *.o *.ofx
	rm -fr GrainPlugin.ofx.bundle

install: GrainPlugin.ofx
	cp -fr GrainPlugin.ofx.bundle /usr/OFX/Plugins
