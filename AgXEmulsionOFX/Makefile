UNAME_SYSTEM := $(shell uname -s)
CXX = g++
CXXFLAGS = -I../ref/grain_ofx/OpenFX-1.4/include -I../ref/grain_ofx/Support/include -Wall -Wextra

ifeq ($(UNAME_SYSTEM), Linux)
	CXXFLAGS += -fPIC -std=gnu++11 -I${CUDAPATH}/include
	CUDAPATH ?= /usr/local/cuda
	NVCC = $(CUDAPATH)/bin/nvcc
	NVCCFLAGS = --compiler-options="-fPIC" -use_fast_math
	LDFLAGS = -shared -L${CUDAPATH}/lib64 -lcuda -lcudart
	BUNDLE_DIR = AgXEmulsionPlugin.ofx.bundle/Contents/Linux-x86-64/
	CUDA_OBJ = SimpleKernel.o
endif

AgXEmulsionPlugin.ofx: AgXEmulsionPlugin.o ${CUDA_OBJ} ofxsCore.o ofxsImageEffect.o ofxsInteract.o ofxsLog.o ofxsMultiThread.o ofxsParams.o ofxsProperty.o ofxsPropertyValidation.o
	$(CXX) $^ -o $@ $(LDFLAGS)
	mkdir -p $(BUNDLE_DIR)
	cp AgXEmulsionPlugin.ofx $(BUNDLE_DIR)

SimpleKernel.o: SimpleKernel.cu
	${NVCC} -c $< $(NVCCFLAGS)

%.o: ../ref/grain_ofx/Support/Library/%.cpp
	$(CXX) -c $< $(CXXFLAGS)

clean:
	rm -f *.o *.ofx
	rm -fr AgXEmulsionPlugin.ofx.bundle

install: AgXEmulsionPlugin.ofx
	cp -fr AgXEmulsionPlugin.ofx.bundle /usr/OFX/Plugins 