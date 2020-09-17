#include <cstdio>
#include <cstring>
#include <cassert>
#include <cmath>
#include <limits>
#include <chrono>

#include <cuda_runtime.h>
#include "helper.cuh"

#define SIZE_T_SENTINEL (std::numeric_limits<size_t>::max())

__global__ void k_szokereses_1gpu(
        size_t *res,
        char const *szo, size_t szo_len,
        char const *mondat, size_t mondat_len) {
    // 1CPU impl masolata
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

    // Siman lehet, hogy ehhez a szalhoz nem tartozik a mondatban karakter
    // Az ilyen szalat nem hagyjuk futni.
    if(base >= mondat_len - szo_len) {
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

    // Siman lehet, hogy ehhez a szalhoz nem tartozik a mondatban karakter
    // Az ilyen szalat nem hagyjuk futni.
    if(x >= mondat_len - szo_len) {
        return;
    }

    // Siman lehet, hogy ehhez a szalhoz nem tartozik a szoban karakter
    // Az ilyen szalat nem hagyjuk futni.
    if(y >= szo_len) {
        return;
    }

    // Ha a karakterek nem egyeznek, lehuzzuk nullara az eredmeny tombben az flaget.
    if(szo[y] != mondat[x + y]) {
        res[x] = 0;
    }
}

// 1-szalas CPU implementacio
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
    int *d_res;
    // thread_dim: szukseges parhuzamossag az X-dimenzioban
    auto thread_dim = mondat_len - szo_len + 1;
    // result_len: az eredmenytomb hossza bajtokban
    auto result_len = thread_dim * sizeof(int);
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

    // Gridet meg nem tanultuk, de eleg nagy bemeneteknel elerem a block meret limitet :(
    // Roviden: mivel a blokkmeret max 1024, ezert 1024x1-es blokkot hasznalunk es
    // a grid meretet ehhez igazitjuk.
    // Mivel a grid mereteinek legalabb 1-nek kell lennie, ezert felfele kerekitunk.
    size_t block_size = 1024;
    size_t grid_size = ceil(thread_dim / (double)block_size);

    // Kernel dispatch
    k_szokereses_nm1gpu<<<grid_size, block_size>>>(d_res, d_szo, szo_len, d_mondat, mondat_len);

    auto kernel_failed = cudaPeekAtLastError() != 0;
    if(kernel_failed) {
        printf("nm1gpu rc=%d\n", cudaGetLastError());
    }

    // Eredmenyt visszamasoljuk, adatokat felszabaditjuk
    cudaMemcpy(h_res, d_res, result_len, cudaMemcpyDeviceToHost);
    cudaFree(d_res);
    cudaFree(d_mondat);
    cudaFree(d_szo);

    if(!kernel_failed) {
        // Ha sikeresen lefutott a program, akkor kikeressuk az elso talalatot.
        for(size_t i = 0; i < thread_dim; i++) {
            // Ahol matcheltunk, oda egyest irtunk.
            if(h_res[i] == 1) {
                *h_idx = i;
                ret = true;
                break;
            }
        }
    }

    return ret;
}

// M*(N-M+1) implementacio
static bool szokereses_mxnm1gpu(
        size_t *h_idx,
        char const *h_szo, size_t szo_len,
        char const *h_mondat, size_t mondat_len) {
    char *d_szo, *d_mondat;
    int *d_res;
    // thread_dim: szukseges parhuzamossag az X-dimenzioban
    auto thread_dim = mondat_len - szo_len + 1;
    // result_len: az eredmenytomb hossza bajtokban
    auto result_len = thread_dim * sizeof(int);
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

    // Gridet meg nem tanultuk, de eleg nagy bemeneteknel elerem a block meret limitet :(
    // Roviden: mivel a blokkmeret max 1024, ezert 512x2-es blokkot hasznalunk es
    // a grid meretet ehhez igazitjuk.
    // Mivel a grid mereteinek legalabb 1-nek kell lennie, ezert felfele kerekitunk.
    auto block_size_x = 512;
    auto block_size_y = 2;
    auto grid_size_x = (long long)ceil(thread_dim / (double)block_size_x);
    auto grid_size_y = (long long)ceil(szo_len / (double)block_size_y);

    dim3 grid_size(grid_size_x, grid_size_y);
    dim3 block_size(block_size_x, block_size_y);

    // Kernel dispatch
    k_szokereses_mxnm1gpu<<<grid_size, block_size>>>(d_res, d_szo, szo_len, d_mondat, mondat_len);

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

// Argumentum feldolgozo
struct arg_decl {
    char const *flag;
    char const **outparam;
};

static bool parse_args(
        int argc,
        char **argv,
        arg_decl const *args) {
    bool ret = true;

    arg_decl const *ad = &args[0];
    while(ad->flag != NULL) {
        *ad->outparam = NULL;
        ad++;
    }

    for(int i = 1; i < argc; i++) {
        ad = &args[0];
        while(ret && ad->flag != NULL) {
            if(strcmp(argv[i], ad->flag) == 0) {
                if(i + 1 < argc) {
                    *ad->outparam = argv[i + 1];
                    i++;
                    break;
                } else {
                    ret = false;
                }
            }

            ad++;
        }

        if(ad->flag == NULL) {
            ret = false;
        }
    }

    return ret;
}

// Program entry
int main(int argc, char **argv) {
    char const *szo = "szo";
    char const *mondat = "asziszoo";
    int reps = 1;
    char const *a_szo;
    char const *a_mondat;
    char const *a_fajl;
    char const *a_reps;

    // Argumentumok feldolgozasa
    arg_decl const argdecls[] = {
        { "-sz", &a_szo },
        { "-m", &a_mondat },
        { "-f", &a_fajl },
        { "-r", &a_reps },
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

        if(a_reps != NULL) {
            sscanf(a_reps, "%d", &reps);
        }
    } else {
        printf("Hasznalat: %s [-sz szo] [-m mondat | -f fajl] [-r reps]\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Kulonbozo implementaciok elinditasa
    auto mondat_len = strlen(mondat);
    auto szo_len = strlen(szo);
    for(int i = 0; i < reps; i++) {
        exec_impl("1CPU",           szo, szo_len, mondat, mondat_len, &szokereses_1cpu);
        exec_impl("1GPU",           szo, szo_len, mondat, mondat_len, &szokereses_1gpu);
        exec_impl("N-M+1 GPU",      szo, szo_len, mondat, mondat_len, &szokereses_nm1gpu);
        exec_impl("M*(N-M+1) GPU",  szo, szo_len, mondat, mondat_len, &szokereses_mxnm1gpu);
    }

    return 0;
}
