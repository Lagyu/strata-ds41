// dmabench: pinned host -> device copy bandwidth on 1 / 2 / 4 streams (does the card have more than one usable copy engine?)
//   hipcc -O2 -o dmabench dmabench.cpp && HIP_VISIBLE_DEVICES=0 ./dmabench [chunk_mb=15] [total_gb=8]
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e)); std::exit(1); } } while (0)
int main(int argc, char** argv) {
    const size_t chunk = (size_t) (argc > 1 ? std::atoi(argv[1]) : 15) << 20;
    const size_t total = (size_t) (argc > 2 ? std::atof(argv[2]) : 8.0) * (1ull << 30);
    const int n = (int) (total / chunk);
    const size_t hbytes = (size_t) 64 * chunk;   // a 64-chunk pinned ring (bigger than any cache)
    uint8_t* h; CK(hipHostMalloc((void**) &h, hbytes, hipHostMallocDefault));
    for (size_t i = 0; i < hbytes; i += 4096) h[i] = (uint8_t) i;
    uint8_t* d; CK(hipMalloc((void**) &d, (size_t) 8 * chunk));
    for (int ns : {1, 2, 4}) {
        std::vector<hipStream_t> s(ns);
        for (auto& x : s) CK(hipStreamCreateWithFlags(&x, hipStreamNonBlocking));
        CK(hipDeviceSynchronize());
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n; ++i)
            CK(hipMemcpyAsync(d + (size_t) (i % 8) * chunk, h + (size_t) (i % 64) * chunk, chunk, hipMemcpyHostToDevice, s[i % ns]));
        CK(hipDeviceSynchronize());
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("H2D %d stream(s): %.1f GB/s (%d x %zu MB)\n", ns, (double) n * chunk / sec / 1e9, n, chunk >> 20);
        for (auto& x : s) CK(hipStreamDestroy(x));
    }
    // D2H for reference
    {
        hipStream_t s; CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n; ++i) CK(hipMemcpyAsync(h + (size_t) (i % 64) * chunk, d + (size_t) (i % 8) * chunk, chunk, hipMemcpyDeviceToHost, s));
        CK(hipDeviceSynchronize());
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("D2H 1 stream: %.1f GB/s\n", (double) n * chunk / sec / 1e9);
    }
    return 0;
}
