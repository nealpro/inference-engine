#include <cuda_runtime.h>

extern "C" __global__ void inference_engine_cuda_probe_kernel(int *out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = 1;
    }
}

extern "C" int inference_engine_cuda_probe(void) {
    int host = 0;
    int *device = nullptr;
    cudaError_t status = cudaMalloc(reinterpret_cast<void **>(&device), sizeof(int));
    if (status != cudaSuccess) return static_cast<int>(status);

    status = cudaMemset(device, 0, sizeof(int));
    if (status == cudaSuccess) {
        inference_engine_cuda_probe_kernel<<<1, 1>>>(device);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) status = cudaMemcpy(&host, device, sizeof(int), cudaMemcpyDeviceToHost);
    cudaError_t free_status = cudaFree(device);
    if (status != cudaSuccess) return static_cast<int>(status);
    if (free_status != cudaSuccess) return static_cast<int>(free_status);
    return host == 1 ? 0 : 9999;
}
