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

// 1-szalas CPU implementacio
static int impl_1cpu(int N, int const* arr) {
    assert(arr != NULL);
    assert(N > 0);

    int m = arr[0];

    for(int i = 1; i < N; i++) {
        if(arr[i] < m) {
            m = arr[i];
        }
    }

    return m;
}

static __global__ void k_min_1gpu(int N, int *d_arr) {
    int m = d_arr[0];

    for(int i = 1; i < N; i++) {
        m = min(m, d_arr[i]);
    }

    d_arr[0] = m;
}

static int impl_1gpu(int N, int const *h_arr) {
    int *d_arr;
    cudaError_t rc;
    int res;

    rc = cudaMalloc(&d_arr, N * sizeof(int));
    rc = cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    k_min_1gpu<<<1, 1>>>(N, d_arr);

    rc = cudaMemcpy(&res, d_arr, sizeof(int), cudaMemcpyDeviceToHost);

    rc = cudaFree(d_arr);

    cudaDeviceSynchronize();

    return res;
}

static __global__ void k_min_ngpu_atomic(int N, int *d_out, int const *d_arr) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= N) {
        return;
    }

    atomicMin(d_out, d_arr[id]);
}

static int impl_ngpu_atomic(int N, int const *h_arr) {
    int *d_arr, *d_out;
    cudaError_t rc;
    int res;

    rc = cudaMalloc(&d_arr, N * sizeof(int));
    rc = cudaMalloc(&d_out, sizeof(int));
    rc = cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);
    rc = cudaMemcpy(d_out, d_arr, sizeof(int), cudaMemcpyDeviceToDevice);

    size_t block_size = 1024;
    size_t grid_size = (N - 1) / block_size + 1;
    k_min_ngpu_atomic<<<grid_size, block_size>>>(N, d_out, d_arr);

    rc = cudaMemcpy(&res, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    rc = cudaFree(d_out);
    rc = cudaFree(d_arr);

    cudaDeviceSynchronize();

    return res;
}

static __global__ void k_min_ngpu_reduce(int N, int *d_arr) {
    int id = (blockIdx.x * blockDim.x + threadIdx.x);

    if(2 * id + 1 >= N) {
        return;
    }

    auto remains = blockDim.x;

    while(remains > 1) {
        auto l = d_arr[2 * id + 0];
        auto r = d_arr[2 * id + 1];
        __syncthreads();

        auto res = min(l, r);
        d_arr[id] = res;
        __syncthreads();

        remains /= 2;
    }
}

static __global__ void k_min_ngpu_reduce_shmem(int N, int *d_arr) {
    int id = (blockIdx.x * blockDim.x + threadIdx.x);

    if(2 * id + 1 >= N) {
        return;
    }

    extern int __shared__ shmem[];

    shmem[2 * threadIdx.x + 0] = d_arr[2 * id + 0];
    shmem[2 * threadIdx.x + 1] = d_arr[2 * id + 0];

    auto remains = blockDim.x;

    while(remains > 1) {
        // auto l = d_arr[2 * id + 0];
        // auto r = d_arr[2 * id + 1];
        auto l = shmem[2 * threadIdx.x + 0];
        auto r = shmem[2 * threadIdx.x+ 1];
        __syncthreads();

        auto res = min(l, r);
        // d_arr[id] = res;
        shmem[threadIdx.x] = res;
        __syncthreads();

        remains /= 2;
    }

    if(threadIdx.x == 0) d_arr[id] = shmem[threadIdx.x];
}

static __global__ void k_collect(int N, int reduceGridDim, int *d_arr) {
    int id = (blockIdx.x * blockDim.x + threadIdx.x);

    if(id >= N) {
        return;
    }

    d_arr[id] = d_arr[id * reduceGridDim];
}

static int impl_ngpu_reduce(int N, int const *h_arr) {
    int *d_arr, *d_out;
    cudaError_t rc;
    int res = 0;

    rc = cudaMalloc(&d_arr, N * sizeof(int));
    rc = cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    auto remains = N;
    while(remains > 1) {
        size_t block_size = 1024;
        size_t grid_size = ((remains / 2) - 1) / block_size + 1;
        k_min_ngpu_reduce<<<grid_size, block_size>>>(remains, d_arr);

        auto collect_block_size = 1024;
        auto collect_grid_size = (grid_size - 1) / collect_block_size + 1;
        k_collect<<<collect_grid_size, collect_block_size>>>(grid_size, block_size, d_arr);

        remains /= 2;
    }

    rc = cudaMemcpy(&res, d_arr, sizeof(int), cudaMemcpyDeviceToHost);

    rc = cudaFree(d_arr);

    cudaDeviceSynchronize();

    return res;
}

static int impl_ngpu_reduce_shmem(int N, int const *h_arr) {
    int *d_arr, *d_out;
    cudaError_t rc;
    int res = 0;

    rc = cudaMalloc(&d_arr, N * sizeof(int));
    rc = cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    auto remains = N;
    while(remains > 1) {
        size_t block_size = 1024;
        size_t grid_size = ((remains / 2) - 1) / block_size + 1;
        auto shmem_bytes = 2 * block_size * sizeof(int);
        k_min_ngpu_reduce_shmem<<<grid_size, block_size, shmem_bytes>>>(remains, d_arr);

        auto collect_block_size = 1024;
        auto collect_grid_size = (grid_size - 1) / collect_block_size + 1;
        k_collect<<<collect_grid_size, collect_block_size>>>(grid_size, block_size, d_arr);

        remains /= 2;
    }

    rc = cudaMemcpy(&res, d_arr, sizeof(int), cudaMemcpyDeviceToHost);

    rc = cudaFree(d_arr);

    cudaDeviceSynchronize();

    return res;
}

// Vegrehajt egy implementaciot es kiirja az eredmenyeket
static void exec_impl(
        char const *nev,
        int N, int const *arr,
        int (*pfunc)(int, int const*)
        ) {
    printf("=== %s ===\n", nev);

    auto t_start = std::chrono::system_clock::now();
    auto res = pfunc(N, arr);
    auto t_end = std::chrono::system_clock::now();
    auto t_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start);

    if(g_printtime) {
        printf("Elapsed: %zu ms\n", t_elapsed.count());
    }

    printf("OK %zu\n", res);
}

// Program entry
int main(int argc, char **argv) {
    long a_reps = 1;
    srand(0);

    // Argumentumok feldolgozasa
    arg_decl const argdecls[] = {
        { "-r", &a_reps, ARG_LONG },
        { "-t", &g_printtime, ARG_LONG },
        { NULL, NULL },
    };

    if(!parse_args(argc, argv, argdecls)) {
        printf("Hasznalat: %s [-r reps]\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Bemenet generalasa
    auto N = 65536 * 16;
    auto arr = new int[N];
    for(int i = 0; i < N; i++) {
        arr[i] = rand();
    }

    // Kulonbozo implementaciok elinditasa
    for(int i = 0; i < a_reps; i++) {
        exec_impl("1CPU",               N, arr, &impl_1cpu);
        exec_impl("1GPU",               N, arr, &impl_1gpu);
        exec_impl("NGPU atomic",        N, arr, &impl_ngpu_atomic);
        exec_impl("NGPU reduce",        N, arr, &impl_ngpu_reduce);
        exec_impl("NGPU reduce shared", N, arr, &impl_ngpu_reduce_shmem);
    }

    return 0;
}
