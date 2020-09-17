NVCC=/usr/local/cuda/bin/nvcc
NVFLAGS=-O2 -ccbin cuda-g++ -Xcompiler -rdynamic -lineinfo -Xcompiler -g -Xcompiler -O2 -g

all: 1_1.exe 1_2.exe 1_3.exe 2.exe

%.exe: %.cu
	$(NVCC) $(NVFLAGS) -o $@ $<

clean:
	rm -f *.exe

.PHONY: all clean
