#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

static int const A[] = { 1, 2, 3, 4, 5 };
static __device__ int dev_A[5];

__global__ void szorzas_single_threaded() {
    for(int i = 0; i < 5; i++) {
        dev_A[i] *= 2;
    }
}

__global__ void szorzas(int mennyivel) {
    int i = threadIdx.x;
    dev_A[i] *= 2;
}

static void print_results(int res[5], char const *title) {
    printf("%s\n", title);
    for(int i = 0; i < 5; i++) {
        printf("A[%d] : %d\n", i, res[i]);
    }
}

int main(int argc, char **argv) {
    int buf[5];

    cudaMemcpyToSymbol(dev_A, A, 5 * sizeof(int));
    szorzas_single_threaded<<<1, 1>>>();
    cudaMemcpyFromSymbol(buf, dev_A, 5 * sizeof(int));
    print_results(buf, "Single-threaded");

    cudaMemcpyToSymbol(dev_A, A, 5 * sizeof(int));
    szorzas<<<1, 5>>>(2);
    cudaMemcpyFromSymbol(buf, dev_A, 5 * sizeof(int));
    print_results(buf, "Multi-threaded");

    return 0;
}
