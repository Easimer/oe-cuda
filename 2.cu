#include <cstdio>
#include <cstring>
#include <cassert>
#include <cmath>
#include <limits>
#include <chrono>

#include <cuda_runtime.h>

#define SIZE_T_SENTINEL (std::numeric_limits<size_t>::max())

static bool szokereses_1cpu(
        size_t *res,
        char const *szo, size_t szo_len,
        char const *mondat, size_t mondat_len) {
    assert(res != NULL);
    assert(szo_len <= mondat_len);

    auto base_max = mondat_len - szo_len + 1;
    for(size_t base = 0; base < base_max; base++) {
        bool found = true;
        for(size_t off = 0; off < szo_len; off++) {
            found &= (mondat[base + off] == szo[off]);
        }

        if(found) {
            *res = base;
            return true;
        }
    }

    return false;
}

__global__ void k_szokereses_1gpu(
        size_t *res,
        char const *szo, size_t szo_len,
        char const *mondat, size_t mondat_len) {
    auto base_max = mondat_len - szo_len + 1;
    for(size_t base = 0; base < base_max; base++) {
        bool found = true;
        for(size_t off = 0; off < szo_len; off++) {
            found &= (mondat[base + off] == szo[off]);
        }

        if(found) {
            *res = base;
            return;
        }
    }
}

__global__ void k_szokereses_nm1gpu(
        int *res,
        char const *szo, size_t szo_len,
        char const *mondat, size_t mondat_len) {
    auto base = blockDim.x * blockIdx.x + threadIdx.x;
    if(base > mondat_len - szo_len) {
        res[base] = 0;
        return;
    }

    for(size_t i = 0; i < szo_len; i++) {
        if(mondat[base + i] != szo[i]) {
            res[base] = 0;
            break;
        }
    }
}

__global__ void k_szokereses_mxnm1gpu(
        int *res,
        char const *szo, size_t szo_len,
        char const *mondat, size_t mondat_len) {
    auto x = blockDim.x * blockIdx.x + threadIdx.x;
    auto y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x >= mondat_len - szo_len) {
        return;
    }

    if(y >= szo_len) {
        return;
    }

    if(szo[y] != mondat[x + y]) {
        res[x] = 0;
    }
}

// 1-szalas GPU implementacio
static bool szokereses_1gpu(
        size_t *h_res,
        char const *h_szo, size_t szo_len,
        char const *h_mondat, size_t mondat_len) {
    char *d_szo, *d_mondat;
    size_t *d_res;
    cudaMalloc(&d_szo, szo_len);
    cudaMalloc(&d_mondat, mondat_len);
    cudaMalloc(&d_res, sizeof(size_t));

    *h_res = SIZE_T_SENTINEL;

    cudaMemcpy(d_szo, h_szo, szo_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mondat, h_mondat, mondat_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, h_res, sizeof(size_t), cudaMemcpyHostToDevice);

    k_szokereses_1gpu<<<1, 1>>>(d_res, d_szo, szo_len, d_mondat, mondat_len);

    cudaMemcpy(h_res, d_res, sizeof(size_t), cudaMemcpyDeviceToHost);

    cudaFree(d_res);
    cudaFree(d_mondat);
    cudaFree(d_szo);

    return *h_res != SIZE_T_SENTINEL;
}

// N-M+1 implementacio
static bool szokereses_nm1gpu(
        size_t *h_idx,
        char const *h_szo, size_t szo_len,
        char const *h_mondat, size_t mondat_len) {
    char *d_szo, *d_mondat;
    int *h_res, *d_res;
    auto thread_dim = mondat_len - szo_len + 1;
    auto result_len = thread_dim * sizeof(int);
    bool ret = false;

    cudaMalloc(&d_szo, szo_len);
    cudaMalloc(&d_mondat, mondat_len);
    cudaMalloc(&d_res, result_len);
    // ha mar muszaj mallocolni a hoston is, akkor pinneljuk azt a szart
    cudaMallocHost(&h_res, result_len);

    // inicializaljuk a result tombot
    for(size_t i = 0; i < thread_dim; i++) {
        h_res[i] = 1;
    }

    cudaMemcpy(d_szo, h_szo, szo_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mondat, h_mondat, mondat_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, h_res, result_len, cudaMemcpyHostToDevice);

    size_t block_size = 1024;
    size_t grid_size = ceil(thread_dim / (double)block_size);

    // Gridet meg nem tanultuk, de eleg nagy bemeneteknel elerem a block meret limitet :(
    printf("kernel<<<%zu, %zu>>>()\n", grid_size, block_size);
    k_szokereses_nm1gpu<<<grid_size, block_size>>>(d_res, d_szo, szo_len, d_mondat, mondat_len);
    auto kernel_failed = cudaPeekAtLastError() != 0;
    printf("nm1gpu rc=%d\n", cudaPeekAtLastError());

    cudaMemcpy(h_res, d_res, result_len, cudaMemcpyDeviceToHost);
    cudaFree(d_res);
    cudaFree(d_mondat);
    cudaFree(d_szo);

    for(size_t i = 0; i < thread_dim; i++) {
        if(h_res[i] != 0) {
            *h_idx = i;
            ret = true;
            break;
        }
    }

    cudaFreeHost(h_res);

    return ret;
}

// M*(N-M+1) implementacio
static bool szokereses_mxnm1gpu(
        size_t *h_idx,
        char const *h_szo, size_t szo_len,
        char const *h_mondat, size_t mondat_len) {
    char *d_szo, *d_mondat;
    int *h_res, *d_res;
    auto thread_dim = mondat_len - szo_len + 1;
    auto result_len = thread_dim * sizeof(int);
    bool ret = false;

    cudaMalloc(&d_szo, szo_len);
    cudaMalloc(&d_mondat, mondat_len);
    cudaMalloc(&d_res, result_len);
    // ha mar muszaj mallocolni a hoston is, akkor pinneljuk azt a szart
    cudaMallocHost(&h_res, result_len);

    // inicializaljuk a result tombot
    for(size_t i = 0; i < thread_dim; i++) {
        h_res[i] = 1;
    }

    cudaMemcpy(d_szo, h_szo, szo_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mondat, h_mondat, mondat_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, h_res, result_len, cudaMemcpyHostToDevice);

    auto block_size_x = 512;
    auto block_size_y = 2;
    auto grid_size_x = (long long)ceil(thread_dim / (double)block_size_x);
    auto grid_size_y = (long long)ceil(szo_len / (double)block_size_y);

    dim3 grid_size(grid_size_x, grid_size_y);
    dim3 block_size(block_size_x, block_size_y);

    // Gridet meg nem tanultuk, de eleg nagy bemeneteknel elerem a block meret limitet :(
    printf("kernel<<<(%zu, %zu), (%zu, %zu)>>>()\n", grid_size_x, grid_size_y, block_size_x, block_size_y);
    k_szokereses_mxnm1gpu<<<grid_size, block_size>>>(d_res, d_szo, szo_len, d_mondat, mondat_len);
    auto kernel_failed = cudaPeekAtLastError() != 0;
    printf("mxnm1gpu rc=%d\n", cudaPeekAtLastError());

    cudaMemcpy(h_res, d_res, result_len, cudaMemcpyDeviceToHost);
    cudaFree(d_res);
    cudaFree(d_mondat);
    cudaFree(d_szo);

    if(!kernel_failed) {
        for(size_t i = 0; i < thread_dim; i++) {
            if(h_res[i] != 0) {
                *h_idx = i;
                ret = true;
                break;
            }
        }
    }

    cudaFreeHost(h_res);

    return ret;
}

// Kiirja a szoveget, de a [from, to[ alszekvencia az magenta lesz
static void print_subseq_bold(char const *s, size_t len, size_t from, size_t to) {
    assert(s != NULL && from <= to);

    size_t start = 0;

    if(from > 50) {
        start = from - 25;

        printf("[...] ");
    }

    for(size_t i = start; i < from; i++) {
        printf("%c", s[i]);
    }

    if(to - from > 0) {
        printf("\033[35m");

        for(size_t i = from; i < to; i++) {
            printf("%c", s[i]);
        }

        printf("\033[m");
    }

    size_t until = len;
    if(until - to > 50) {
        until = to + 25;
    }

    for(size_t i = to; i < until; i++) {
        printf("%c", s[i]);
    }

    if(until != len) {
        printf(" [...]");
    }
}

static void exec_impl(
        char const *nev,
        char const *szo, size_t szo_len,
        char const *mondat, size_t mondat_len,
        bool (*pfunc)(size_t*, char const*, size_t, char const*, size_t)
        ) {
    bool rc;
    size_t idx;

    printf("=== %s ===\n", nev);

    auto t_start = std::chrono::system_clock::now();
    rc = pfunc(&idx, szo, szo_len, mondat, mondat_len);
    auto t_end = std::chrono::system_clock::now();
    auto t_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start);

    printf("Elapsed: %zu ms\n", t_elapsed.count());

    if(rc) {
        printf("? '%s' \\in '", szo);
        print_subseq_bold(mondat, mondat_len, idx, idx + szo_len);
        printf("'\n");

        printf("OK %zu\n", idx);
    } else {
        //printf("? '%s' \\in '%s'\n", szo, mondat);
        printf("FAIL\n");
    }
}

static char const* load_src(char const *path) {
    FILE* f = fopen(path, "rb");
    if(f == NULL) {
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    auto len = ftell(f);
    rewind(f);
    char *buf = NULL;
    cudaMallocHost(&buf, len + 1);

    fread(buf, 1, len, f);
    fclose(f);

    return buf;
}

int main(int argc, char **argv) {
    char const *szo, *mondat;

    if(argc == 4) {
        // ./2.exe szo -c forras.txt
        mondat = load_src(argv[3]);
        if(mondat == NULL) {
            fprintf(stderr, "Nem sikerult kiolvasni: '%s'\n", argv[3]);
            return EXIT_FAILURE;
        }
        szo = argv[1];
    } else if(argc == 3 || argc == 1) {
        // ./2.exe szo mondat 
        szo =
            (argc > 1 && argv[1]) ? argv[1] : "szo";
        mondat =
            (argc > 2 && argv[2]) ? argv[2] : "asziszoo";
    } else {
        return EXIT_FAILURE;
    }

    for(int i = 0; i < 16; i++) {
        exec_impl("1CPU", szo, strlen(szo), mondat, strlen(mondat), &szokereses_1cpu);
        // exec_impl("1GPU", szo, strlen(szo), mondat, strlen(mondat), &szokereses_1gpu);
        exec_impl("N-M+1 GPU", szo, strlen(szo), mondat, strlen(mondat), &szokereses_nm1gpu);
        exec_impl("M*(N-M+1) GPU", szo, strlen(szo), mondat, strlen(mondat), &szokereses_mxnm1gpu);
    }

    return 0;
}
