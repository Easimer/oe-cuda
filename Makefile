NVCC=/usr/local/cuda/bin/nvcc
NVFLAGS=-ccbin cuda-g++ -Xcompiler -rdynamic -lineinfo -Xcompiler -g -g

all: 1_1.exe 1_2.exe 1_3.exe 2.exe 4_1.exe 4_2.exe 6_2.exe

%.exe: %.cu
	$(NVCC) $(NVFLAGS) -o $@ $<

clean:
	rm -f *.exe

.PHONY: all clean
