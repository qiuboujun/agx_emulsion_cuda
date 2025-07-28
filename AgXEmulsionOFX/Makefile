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

AgXEmulsionPlugin.ofx: AgXEmulsionPlugin.o ${CUDA_OBJ} ${LOCAL_OBJ} ofxsCore.o ofxsImageEffect.o ofxsInteract.o ofxsLog.o ofxsMultiThread.o ofxsParams.o ofxsProperty.o ofxsPropertyValidation.o
	$(CXX) $^ -o $@ $(LDFLAGS)
	mkdir -p $(BUNDLE_DIR)
	cp AgXEmulsionPlugin.ofx $(BUNDLE_DIR)

SimpleKernel.o: SimpleKernel.cu
	${NVCC} -c $< $(NVCCFLAGS)

%.o: ../ref/grain_ofx/Support/Library/%.cpp
	$(CXX) -c $< $(CXXFLAGS)

# compile local cpp
%.o: %.cpp
	$(CXX) -c $< $(CXXFLAGS)

clean:
	rm -f *.o *.ofx
	rm -fr AgXEmulsionPlugin.ofx.bundle

install: AgXEmulsionPlugin.ofx
	cp -fr AgXEmulsionPlugin.ofx.bundle /usr/OFX/Plugins 

# ---------- unit tests -----------
.PHONY: tests
tests: tests/test_illuminants tests/test_density tests/test_density_spline tests/test_parametric tests/test_couplers tests/test_emulsion tests/test_emulsion_coupler
	./tests/test_illuminants && ./tests/test_density && ./tests/test_density_spline && ./tests/test_parametric && ./tests/test_couplers && ./tests/test_emulsion && ./tests/test_emulsion_coupler

tests/test_illuminants: ../tests/test_illuminants.cpp ../spectral_helpers.cpp
	mkdir -p tests
	$(CXX) -I.. -std=gnu++11 $^ -o $@

tests/test_density: ../tests/test_density.cpp ../density_curves.cpp
	mkdir -p tests
	$(CXX) -I.. -std=gnu++11 $^ -o $@ 

tests/test_density_spline: ../tests/test_density_spline.cpp ../density_spline.cpp
	mkdir -p tests
	$(CXX) -I.. -std=gnu++11 $^ -o $@ 

tests/prep_parametric:
	python ../tests/gen_parametric_reference.py

tests/test_parametric: tests/prep_parametric ../tests/test_parametric.cpp ../parametric.cpp
	mkdir -p tests
	$(CXX) -I.. -std=gnu++11 ../tests/test_parametric.cpp ../parametric.cpp -o $@ 

tests/prep_couplers:
	python ../tests/gen_couplers_reference.py

tests/test_couplers: tests/prep_couplers ../tests/test_couplers.cpp ../couplers.cpp
	mkdir -p tests
	$(CXX) -I.. -std=gnu++11 ../tests/test_couplers.cpp ../couplers.cpp -o $@ 

tests/prep_emulsion:
	python ../tests/gen_emulsion_reference.py

tests/test_emulsion: tests/prep_emulsion ../tests/test_emulsion.cpp ../emulsion.cpp ../interpolate.cpp ../couplers.cpp
	mkdir -p tests
	$(CXX) -I.. -std=gnu++11 ../tests/test_emulsion.cpp ../emulsion.cpp ../interpolate.cpp ../couplers.cpp -o $@ 

tests/test_emulsion_coupler: ../tests/test_emulsion_coupler.cpp ../emulsion.cpp ../interpolate.cpp ../couplers.cpp
	mkdir -p tests
	$(CXX) -I.. -std=gnu++11 ../tests/test_emulsion_coupler.cpp ../emulsion.cpp ../interpolate.cpp ../couplers.cpp -o $@ 