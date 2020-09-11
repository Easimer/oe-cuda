NVCC=/usr/local/cuda/bin/nvcc
NVFLAGS=-ccbin cuda-g++

%.o: %.cu
	$(NVCC) $(NVFLAGS) -o $@ -dc $<

1_1.exe: 1_1.o
	$(NVCC) $(NVFLAGS) -o 1_1.exe 1_1.o

all: *.exe
