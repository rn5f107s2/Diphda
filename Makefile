SRC_DIR=src

SOURCES := $(shell find src -name '*.cpp')
SOURCES += $(shell find src -name '*.cu')

all:
	nvcc $(SOURCES) -o Diphda -lcudnn -arch=sm_80 -std=c++20 -O3 -Xcompiler -fopenmp
