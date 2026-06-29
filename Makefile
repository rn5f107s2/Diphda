SRC_DIR=src

SOURCES := $(shell find src -name '*.cpp')
SOURCES += $(shell find src -name '*.cu')

all:
	nvcc $(SOURCES) -o Diphda -lcudnn -std=c++20 -O3
