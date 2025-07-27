UNAME_SYSTEM := $(shell uname -s)
CXX          = g++

# Paths to shared OFX helper library (reuse from grain plugin)
SUPPORT_DIR  := ../ref/grain_ofx/Support
OFX_INCLUDE  := ../ref/grain_ofx/OpenFX-1.4/include

CXXFLAGS     = -I$(OFX_INCLUDE) -I$(SUPPORT_DIR)/include -Wall -Wextra -fPIC -std=gnu++11

ifeq ($(UNAME_SYSTEM), Linux)
    LDFLAGS = -shared
    BUNDLE_DIR = AgXEmulsionOFX.ofx.bundle/Contents/Linux-x86-64/
endif

PLUGIN_SRC   = AgXEmulsionPlugin.cpp
PLUGIN_OBJ   = $(PLUGIN_SRC:.cpp=.o)

SUPPORT_OBJS = ofxsCore.o ofxsImageEffect.o ofxsInteract.o ofxsLog.o \
               ofxsMultiThread.o ofxsParams.o ofxsProperty.o ofxsPropertyValidation.o

PLUGIN_LIB   = AgXEmulsionPlugin.ofx

all: $(PLUGIN_LIB)

$(PLUGIN_LIB): $(PLUGIN_OBJ) $(SUPPORT_OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS)
	# Bundle directory structure
	mkdir -p $(BUNDLE_DIR)
	cp $@ $(BUNDLE_DIR)

%.o: %.cpp
	$(CXX) -c $< $(CXXFLAGS)

# Compile support library source files from Support/Library
ofxs%.o: $(SUPPORT_DIR)/Library/ofxs%.cpp
	$(CXX) -c $< $(CXXFLAGS)

clean:
	rm -f *.o *.ofx
	rm -rf AgXEmulsionOFX.ofx.bundle

install: all
	cp -fr AgXEmulsionOFX.ofx.bundle /usr/OFX/Plugins 