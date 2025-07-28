UNAME_SYSTEM := $(shell uname -s)
CXX = g++
CXXFLAGS = -I.. -I../ref/grain_ofx/OpenFX-1.4/include -I../ref/grain_ofx/Support/include -Wall -Wextra -std=gnu++11 -fPIC

ifeq ($(UNAME_SYSTEM), Linux)
	CUDAPATH = /usr/local/cuda
	NVCC = $(CUDAPATH)/bin/nvcc
	NVCCFLAGS = --compiler-options="-fPIC" -use_fast_math
	LDFLAGS = -shared -L${CUDAPATH}/lib64 -lcuda -lcudart -Wl,-rpath,'$$ORIGIN'
	BUNDLE_DIR = AgXEmulsionPlugin.ofx.bundle/Contents/Linux-x86-64/
	CUDA_OBJ = EmulsionKernel.o DiffusionHalationKernel.o GrainKernel.o PaperKernel.o
	CXXFLAGS_PLUGIN = -I.. -I../ref/grain_ofx/OpenFX-1.4/include -I../ref/grain_ofx/Support/include -Wall -Wextra -std=gnu++11 -fPIC -I${CUDAPATH}/include
endif

all: AgXEmulsionPlugin.ofx libAgXLUT.so

libAgXLUT.so: ../LUTHelper.cpp
	$(CXX) -shared -std=gnu++11 -fPIC $< -o $@ -Wl,-rpath,'$$ORIGIN'

AgXEmulsionPlugin.ofx: AgXEmulsionPlugin.o ${CUDA_OBJ} ofxsCore.o ofxsImageEffect.o ofxsInteract.o ofxsLog.o ofxsMultiThread.o ofxsParams.o ofxsProperty.o ofxsPropertyValidation.o libAgXLUT.so
	$(CXX) $(filter-out libAgXLUT.so,$^) -o $@ $(LDFLAGS)
	mkdir -p $(BUNDLE_DIR)
	cp AgXEmulsionPlugin.ofx $(BUNDLE_DIR)
	cp libAgXLUT.so $(BUNDLE_DIR)

EmulsionKernel.o: EmulsionKernel.cu
	${NVCC} -c $< $(NVCCFLAGS)

DiffusionHalationKernel.o: DiffusionHalationKernel.cu
	${NVCC} -c $< $(NVCCFLAGS)

GrainKernel.o: GrainKernel.cu
	${NVCC} -c $< $(NVCCFLAGS)

PaperKernel.o: PaperKernel.cu
	${NVCC} -c $< $(NVCCFLAGS)

AgXEmulsionPlugin.o: AgXEmulsionPlugin.cpp
	$(CXX) -c $< $(CXXFLAGS_PLUGIN)

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