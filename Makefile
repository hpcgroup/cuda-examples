CXX = g++
CXXFLAGS = -std=c++11
CUDA = nvcc
CUDAFLAGS = -std=c++11 -Xcompiler "$(CXXFLAGS)"

HOSTNAME = $(shell hostname | sed 's/[0-9]//g')
ifeq ($(HOSTNAME), lassen)
GENCODE=--generate-code arch=compute_70,code=sm_70
endif

CU_FILES = $(wildcard */*.cu)
TARGETS = $(patsubst %.cu,%, $(CU_FILES))

all: $(TARGETS)

%: %.cu
	$(CUDA) $(CUDAFLAGS) $(GENCODE) -o $@ $<

clean:
	rm $(TARGETS)
