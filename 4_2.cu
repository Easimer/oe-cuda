#include <cstdio>
#include <cstring>
#include <cassert>
#include <cmath>
#include <limits>
#include <chrono>

#include <cuda_runtime.h>
#include "helper.cuh"
#include "arg_parse.h"

static long g_printtime = 0;
static long g_printtext = 0;

__global__ void k_rendezes(
        int* g_v, unsigned N
    ) {
    extern __shared__ int smem[];

    if(threadIdx.x > N * 2) {
        return;
    }

    smem[threadIdx.x * 2 + 0] = g_v[threadIdx.x * 2 + 0];
    smem[threadIdx.x * 2 + 1] = g_v[threadIdx.x * 2 + 1];

    int x = threadIdx.x;

    int id0 = 2 * x + 0;
    int id1 = 2 * x + 1;
    int id2 = 2 * x + 2;
    for(int iter = 0; iter < N / 2; iter++) {
        int v0, v1;

        v0 = smem[id0];
        v1 = smem[id1];
        if(v0 > v1) {
            smem[2 * x + 0] = v1;
            smem[2 * x + 1] = v0;
        }
        __syncthreads();

        if(id2 < N) {
            v0 = smem[2 * x + 1];
            v1 = smem[2 * x + 2];
            if(v0 > v1) {
                smem[2 * x + 1] = v1;
                smem[2 * x + 2] = v0;
            }
        }
        __syncthreads();
    }

    g_v[threadIdx.x * 2 + 0] = smem[threadIdx.x * 2 + 0];
    g_v[threadIdx.x * 2 + 1] = smem[threadIdx.x * 2 + 1];
}

__global__ void k_rendezes_global(
        int* g_v, unsigned N
    ) {
    if(threadIdx.x > N * 2) {
        return;
    }

    int x = threadIdx.x;

    int id0 = 2 * x + 0;
    int id1 = 2 * x + 1;
    int id2 = 2 * x + 2;
    for(int iter = 0; iter < N / 2; iter++) {
        int v0, v1;

        v0 = g_v[id0];
        v1 = g_v[id1];
        if(v0 > v1) {
            g_v[2 * x + 0] = v1;
            g_v[2 * x + 1] = v0;
        }
        __syncthreads();

        if(id2 < N) {
            v0 = g_v[2 * x + 1];
            v1 = g_v[2 * x + 2];
            if(v0 > v1) {
                g_v[2 * x + 1] = v1;
                g_v[2 * x + 2] = v0;
            }
        }
        __syncthreads();
    }
}

int main(int argc, char **argv) {
    int* h_p = new int[2048];
    int* d_p;

    cudaMalloc(&d_p, 2048 * sizeof(int));

    for(int r = 0; r < 64; r++) {
        for(int i = 0; i < 2048; i++) h_p[i] = rand() % 64;
        cudaMemcpy(d_p, h_p, 2048 * sizeof(int), cudaMemcpyHostToDevice);
        k_rendezes<<<1, 1024, 2048 * sizeof(int)>>>(d_p, 2048);
        cudaMemcpy(h_p, d_p, 2048 * sizeof(int), cudaMemcpyDeviceToHost);

        for(int i = 1; i < 2048; i++) {
            auto v0 = h_p[i - 1];
            auto v1 = h_p[i - 0];
            if(v0 > v1) {
                printf("ERROR AT IDX %d-%d val=%d:%d\n", i-1, i, v0, v1);
            }
        }

        for(int i = 0; i < 2048; i++) h_p[i] = rand() % 64;
        cudaMemcpy(d_p, h_p, 2048 * sizeof(int), cudaMemcpyHostToDevice);
        k_rendezes_global<<<1, 1024>>>(d_p, 2048);
        cudaMemcpy(h_p, d_p, 2048 * sizeof(int), cudaMemcpyDeviceToHost);

        for(int i = 1; i < 2048; i++) {
            auto v0 = h_p[i - 1];
            auto v1 = h_p[i - 0];
            if(v0 > v1) {
                printf("ERROR AT IDX %d-%d val=%d:%d\n", i-1, i, v0, v1);
            }
        }
    }

    cudaFree(d_p);

    delete[] h_p;
    return 0;
}
