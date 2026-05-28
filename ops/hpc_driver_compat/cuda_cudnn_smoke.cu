#include <cstdio>

#include <cuda_runtime.h>
#include <cudnn.h>

__global__ void add_one(float *data) {
    int i = threadIdx.x;
    data[i] += 1.0f;
}

static void check_cuda(cudaError_t status, const char *what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
        std::exit(2);
    }
}

int main() {
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    std::printf("CUDA_DEVICE_COUNT=%d\n", device_count);
    if (device_count < 1) {
        std::fprintf(stderr, "No CUDA devices visible\n");
        return 3;
    }

    int runtime_version = 0;
    int driver_version = 0;
    check_cuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
    check_cuda(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
    std::printf("CUDA_RUNTIME_VERSION=%d\n", runtime_version);
    std::printf("CUDA_DRIVER_VERSION=%d\n", driver_version);

    cudaDeviceProp prop;
    check_cuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    std::printf("CUDA_DEVICE_0=%s\n", prop.name);
    std::printf("CUDA_DEVICE_0_MEM=%zu\n", prop.totalGlobalMem);

    float host[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float *device = nullptr;
    check_cuda(cudaMalloc(&device, sizeof(host)), "cudaMalloc");
    check_cuda(cudaMemcpy(device, host, sizeof(host), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    add_one<<<1, 4>>>(device);
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    check_cuda(cudaMemcpy(host, device, sizeof(host), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    check_cuda(cudaFree(device), "cudaFree");
    std::printf("CUDA_KERNEL_RESULT=%.1f %.1f %.1f %.1f\n", host[0], host[1], host[2], host[3]);

    std::printf("CUDNN_VERSION=%zu\n", static_cast<size_t>(cudnnGetVersion()));
    cudnnHandle_t handle;
    cudnnStatus_t cudnn_status = cudnnCreate(&handle);
    if (cudnn_status != CUDNN_STATUS_SUCCESS) {
        std::fprintf(stderr, "cudnnCreate failed: %s\n", cudnnGetErrorString(cudnn_status));
        return 4;
    }
    cudnnDestroy(handle);
    std::printf("CUDNN_CREATE=OK\n");
    return 0;
}
