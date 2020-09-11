NVCC=/usr/local/cuda/bin/nvcc
NVFLAGS=-ccbin cuda-g++

all: 1_1.exe 1_2.exe 1_3.exe

%.exe: %.cu
	$(NVCC) $(NVFLAGS) -o $@ $<

clean:
	rm -f *.exe

.PHONY: all clean
