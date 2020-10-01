#include <cstdio>
#include <cstring>
#include <cassert>
#include <cmath>
#include <limits>
#include <chrono>

#include <cuda_runtime.h>
#include "helper.cuh"
#include "arg_parse.h"

#define SIZE_T_SENTINEL (std::numeric_limits<size_t>::max())

static long g_printtime = 0;
static long g_printtext = 0;

__global__ void k_szokereses_mxnm1gpu(
        int *res,
        char const *szo, size_t szo_len,
        char const *mondat, size_t mondat_len) {
    auto x = blockDim.x * blockIdx.x + threadIdx.x;
    auto y = blockDim.y * blockIdx.y + threadIdx.y;

    extern __shared__ char smem[];

    if(x >= mondat_len) {
        return;
    }

    // TODO: nem hasznaljuk
    /*
    auto s_mondat = &smem[blockDim.y];

    if(threadIdx.y == 0) {
        s_mondat[threadIdx.x] = mondat[x];
    }
    __syncthreads();
    */

    // Siman lehet, hogy ehhez a szalhoz nem tartozik a szoban karakter
    // Az ilyen szalat nem hagyjuk futni.
    if(y >= szo_len) {
        return;
    }

    auto s_szo = &smem[0];

    if(threadIdx.x == 0) {
        s_szo[threadIdx.y] = szo[y];
    }

    // Siman lehet, hogy ehhez a szalhoz nem tartozik a mondatban karakter
    // Az ilyen szalat nem hagyjuk futni.
    if(x > mondat_len - szo_len) {
        return;
    }

    __syncthreads();

    // Ha a karakterek nem egyeznek, lehuzzuk nullara az eredmeny tombben az flaget.
    if(s_szo[threadIdx.y] != mondat[x + y]) {
        res[x] = 0;
    }
}

// M*(N-M+1) implementacio
static bool szokereses_mxnm1gpu(
        size_t *h_idx,
        char const *h_szo, size_t szo_len,
        char const *h_mondat, size_t mondat_len) {
    char *d_szo, *d_mondat;
    int *d_res;
    // thread_dim: szukseges parhuzamossag az X-dimenzioban
    auto thread_dim = mondat_len;
    // result_len: az eredmenytomb hossza bajtokban
    auto result_len = mondat_len * sizeof(int);
    bool ret = false;

    cudaMalloc(&d_szo, szo_len);
    cudaMalloc(&d_mondat, mondat_len);
    cudaMalloc(&d_res, result_len);

    // Lefoglalunk memoriat, ami lehet pinnelt, lehet nem (input merettol fugg, hogy a runtime engedi-e)
    Maybe_Pinned<int> h_res(thread_dim);

    // inicializaljuk a result tombot
    for(size_t i = 0; i < thread_dim; i++) {
        h_res[i] = 1;
    }

    // Felmasoljuk az adatokat
    cudaMemcpy(d_szo, h_szo, szo_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mondat, h_mondat, mondat_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, h_res, result_len, cudaMemcpyHostToDevice);

    // Grid meret kiszamitasa
    auto block_size_x = 512;
    auto block_size_y = 2;
    auto shared_mem = block_size_x + block_size_y;
    auto grid_size_x = (thread_dim - 1) / block_size_x + 1;
    auto grid_size_y = (szo_len - 1) / block_size_y + 1;

    dim3 grid_size(grid_size_x, grid_size_y);
    dim3 block_size(block_size_x, block_size_y);

    // Kernel dispatch
    k_szokereses_mxnm1gpu<<<grid_size, block_size, shared_mem>>>(d_res, d_szo, szo_len, d_mondat, mondat_len);

    auto kernel_failed = cudaPeekAtLastError() != 0;
    if(kernel_failed) {
        printf("mxnm1gpu rc=%d\n", cudaGetLastError());
    }

    // Eredmenyt visszamasoljuk, adatokat felszabaditjuk
    cudaMemcpy(h_res, d_res, result_len, cudaMemcpyDeviceToHost);
    cudaFree(d_res);
    cudaFree(d_mondat);
    cudaFree(d_szo);

    if(!kernel_failed) {
        // Ha sikeresen lefutott a program, akkor kikeressuk az elso talalatot.
        for(size_t i = 0; i < thread_dim; i++) {
            // Ahol nincs match, oda nullat irtunk. Tehat nullatol kulonbozo erteket keresunk.
            if(h_res[i] != 0) {
                *h_idx = i;
                ret = true;
                break;
            }
        }
        for(size_t i = 0; i < thread_dim; i++) {
            printf("%d ", h_res[i]);
        }
    }

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

// Vegrehajt egy implementaciot es kiirja az eredmenyeket
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

    if(g_printtime) {
        printf("Elapsed: %zu ms\n", t_elapsed.count());
    }

    if(rc) {
        if(g_printtext) {
            printf("? '%s' \\in '", szo);
            print_subseq_bold(mondat, mondat_len, idx, idx + szo_len);
            printf("'\n");
        }

        printf("OK %zu\n", idx);
    } else {
        printf("FAIL\n");
    }
}

// Betolt egy egesz fajlt pinned memoriaba
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

    auto rd = fread(buf, 1, len, f);
    buf[rd] = '\0';
    buf[len] = '\0';
    fclose(f);

    return buf;
}

// Program entry
int main(int argc, char **argv) {
    char const *szo = NULL;
    char const *mondat = NULL;
    char const *a_szo = "szo";
    char const *a_mondat = "asziszoo";
    char const *a_fajl = NULL;
    long a_reps = 1;

    // Argumentumok feldolgozasa
    arg_decl const argdecls[] = {
        { "-sz", &a_szo, ARG_STRING },
        { "-m", &a_mondat, ARG_STRING },
        { "-f", &a_fajl, ARG_STRING },
        { "-r", &a_reps, ARG_LONG },
        { "-t", &g_printtime, ARG_LONG },
        { "-e", &g_printtext, ARG_LONG },
        { NULL, NULL },
    };

    if(parse_args(argc, argv, argdecls)) {
        if(a_fajl != NULL) {
            mondat = load_src(a_fajl);
            if(mondat == NULL) {
                fprintf(stderr, "Nem sikerult kiolvasni: '%s'\n", argv[3]);
                return EXIT_FAILURE;
            }
        } else if(a_mondat != NULL) {
            mondat = a_mondat;
        }

        if(a_szo != NULL) {
            szo = a_szo;
        }
    } else {
        printf("Hasznalat: %s [-sz szo] [-m mondat | -f fajl] [-r reps] [-t 0/1] [-e 0/1]\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Kulonbozo implementaciok elinditasa
    auto mondat_len = strlen(mondat);
    auto szo_len = strlen(szo);
    for(int i = 0; i < a_reps; i++) {
        exec_impl("M*(N-M+1) GPU",  szo, szo_len, mondat, mondat_len, &szokereses_mxnm1gpu);
    }

    return 0;
}
