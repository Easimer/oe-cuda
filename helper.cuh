#pragma once

// Absztrakcio ami lehetove teszi, hogy ha nem engedi
// a driver, hogy pinneljunk egy memoriat valami okbol,
// akkor fallback-eljunk sima rendszermemoriara.
// Plusz menedzseli az eroforrast.
template<typename T>
class Maybe_Pinned {
    bool is_pinned;
    T* ptr;
public:
    Maybe_Pinned(size_t count) : ptr(NULL), is_pinned(true) {
        cudaError_t rc;
        
        rc = cudaMallocHost(&ptr, count * sizeof(T));

        if(rc != 0) {
            is_pinned = false;
            ptr = new T[count];
        }
    }

    ~Maybe_Pinned() {
        if(ptr != NULL) {
            if(is_pinned) {
                cudaFreeHost(ptr);
            } else {
                delete[] ptr;
            }
        }
    }

    // Nincs mozgatas, masolas
    Maybe_Pinned(Maybe_Pinned const&) = delete;
    Maybe_Pinned(Maybe_Pinned&&) = delete;
    void operator=(Maybe_Pinned const&) = delete;
    void operator=(Maybe_Pinned&&) = delete;

    operator T*() const {
        return ptr;
    }
};
