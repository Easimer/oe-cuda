NVCC=/usr/local/cuda/bin/nvcc
NVFLAGS=-ccbin cuda-g++

all: 1_1.exe 1_2.exe

%.o: %.cu
	$(NVCC) $(NVFLAGS) -o $@ -dc $<

1_1.exe: 1_1.o
	$(NVCC) $(NVFLAGS) -o 1_1.exe 1_1.o

1_2.exe: 1_2.o
	$(NVCC) $(NVFLAGS) -o 1_2.exe 1_2.o

clean:
	rm -f *.exe *.o

.PHONY: all clean
