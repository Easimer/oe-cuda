#include <cstdio>

constexpr int N = 128;
constexpr int SHMEM_DIM = 8;

static void fill_rand_matrix(float *d_matrix) {
    cudaError_t rc;
    float buf[32];

    auto total = N * N;
    auto offset = 0;

    rc = cudaHostRegister(buf, 32 * sizeof(float), 0);

    while(total > 0) {
        for(int i = 0; i < 32; i++) {
            buf[i] = ((rand() % 16) - 8) / 2.0f;
        }

        rc = cudaMemcpy(d_matrix + offset, buf, 32 * sizeof(float), cudaMemcpyHostToDevice);
        offset += 32;
        total -= 32;
    }

    if(total > 0) {
        for(int i = 0; i < total; i++) {
            buf[i] = ((rand() % 16) - 8) / 2.0f;
        }

        rc = cudaMemcpy(d_matrix + offset, buf, total * sizeof(float), cudaMemcpyHostToDevice);
    }

    rc = cudaHostUnregister(buf);
}

static void fill_identity(float *d_matrix) {
    float buf[N];

    for(int y = 0; y < N; y++) {
        for(int i = 0; i < N; i++) {
            if(i == y) {
                buf[i] = 1;
            } else {
                buf[i] = 0;
            }
        }

        cudaMemcpy(d_matrix + y * N, buf, N * sizeof(float), cudaMemcpyHostToDevice);
    }
}

static __global__ void k_mat_mul(int N, float *dst, float const *lhs, float const *rhs) {
    extern __shared__ float tile[];
    float *tile0 = tile + 0 * blockDim.x * blockDim.y;
    float *tile1 = tile + 1 * blockDim.x * blockDim.y;

    auto gx = blockDim.x * blockIdx.x + threadIdx.x;
    auto gy = blockDim.y * blockIdx.y + threadIdx.y;
    auto sh_off = threadIdx.y * blockDim.x + threadIdx.x;

    float acc = 0;
    auto y_base = gy * N;
    for(int j = 0; j < gridDim.x; j++) {
        auto block_off = j * blockDim.x;
        auto idx = y_base + block_off + threadIdx.x;
        tile0[sh_off] = lhs[idx];
        tile1[sh_off] = rhs[idx];
        __syncthreads();

        for(int I = 0; I < blockDim.x; I++) {
            acc += tile0[threadIdx.y * blockDim.x + I] * tile1[I * blockDim.x + threadIdx.x];
        }

        __syncthreads();
    }

    dst[gy * N + gx] = acc;
}

static void print_matrix(float *d_matrix) {
    float buf[N];

    for(int y = 0; y < N; y++) {
        cudaMemcpy(buf, d_matrix + y * N, N * sizeof(float), cudaMemcpyDeviceToHost);
        for(int i = 0; i < N; i++) {
            printf("%+6.2f ", buf[i]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    cudaError_t rc;
    float *d_mat0, *d_mat1, *d_mat2;

    srand(0);

    auto total_size = N * N * sizeof(float);

    rc = cudaMalloc(&d_mat0, total_size);
    rc = cudaMalloc(&d_mat1, total_size);
    rc = cudaMalloc(&d_mat2, total_size);

    fill_rand_matrix(d_mat0);
    fill_rand_matrix(d_mat1);
    // fill_identity(d_mat1);

    auto shmem_siz = 2 * SHMEM_DIM * SHMEM_DIM * sizeof(float);
    auto block_size = dim3(SHMEM_DIM, SHMEM_DIM, 1);
    auto g = 1 + (N - 1) / SHMEM_DIM;
    auto grid_size = dim3(g, g, 1);

    k_mat_mul<<<grid_size, block_size, shmem_siz>>>(N, d_mat2, d_mat0, d_mat1);

    print_matrix(d_mat2);

    rc = cudaFree(d_mat2);
    rc = cudaFree(d_mat1);
    rc = cudaFree(d_mat0);
    return 0;
}
