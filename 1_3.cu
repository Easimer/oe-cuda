#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

static void print_results(int *a, int *b, int *c, int N, char const *title) {
    printf("%s\n", title);
    for(int i = 0; i < N; i++) {
        printf("[%d] min(%d, %d) => %d\n", i, a[i], b[i], c[i]);
    }
}

__global__ void kernel_min(int *d_C, int* d_A, int* d_B) {
    int i = threadIdx.x;

    d_C[i] = min(d_A[i], d_B[i]);
}

static void upload_and_exec(int *h_C, int const *h_A, int const *h_B, int N) {
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(int));
    cudaMalloc(&d_B, N * sizeof(int));
    cudaMalloc(&d_C, N * sizeof(int));

    cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice);

    kernel_min<<<1, N>>>(d_C, d_A, d_B);

    cudaMemcpy(h_C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_C);
    cudaFree(d_B);
    cudaFree(d_A);
}

static void fill_rand(int *p, int N) {
    for(int i = 0; i < N; i++) {
        p[i] = rand() % 32;
    }
}

int main(int argc, char **argv) {
#define N (16)
    int h_A[N];
    int h_B[N];
    int h_C[N];

    srand(0);
    fill_rand(h_A, N);
    fill_rand(h_B, N);

    upload_and_exec(h_C, h_A, h_B, N);

    print_results(h_A, h_B, h_C, N, "Sum");

    return 0;
}
