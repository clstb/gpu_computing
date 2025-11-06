#include <filesystem>
#include <random>
#include <iostream>
#include <chrono>

void init(int32_t size, int32_t *vec_a, int32_t *vec_b, int32_t *mat)
{
    // std::random_device dev;
    std::mt19937 prng(2024);
    std::uniform_int_distribution<int32_t> distrib(-16, 16);

    for (auto i = 0; i < size; i++)
    {
        vec_a[i] = distrib(prng);
        vec_b[i] = distrib(prng);
    }

    for (auto i = 0; i < size * size; i++)
        mat[i] = distrib(prng);
}

void compute(int32_t size, int32_t *vec_a, int32_t *vec_b, int32_t *mat, int32_t *out)
{
    auto tmp = (int32_t *)malloc(sizeof(int32_t) * size);
    for (auto i = 0; i < size; i++)
        tmp[i] = vec_a[i] + vec_b[i];

    for (auto i = 0; i < size; i++)
    {
        out[i] = 0;
        for (auto j = 0; j < size; j++)
            out[i] += tmp[j] * mat[i * size + j];
    }
    free(tmp);
}

__inline__ __device__ float warpReduceSum(int32_t val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void hadamard(int32_t size, int32_t *vec_a, int32_t *vec_b, int32_t *mat, int32_t *out) {
    int tid = threadIdx.x;
    if (tid > size) return;
    int32_t row = blockIdx.x;
    int32_t sum = 0;

    // Each thread handles the elements which are a multiple of the block size
    // I.e thread 0 handles elements 0, 256, 512, 1024 assuming max 1024 threads
    // Thread 1 handles elements 1, 257, 513, etc.
    for (int i = tid; i < size; i += blockDim.x)
        sum += (vec_a[i] + vec_b[i]) * mat[row * size + i];

    // Collect the result from every warp group
    sum = warpReduceSum(sum);

    // Checking if the thread is the first of it's warp group
    // Then compute the warp index by tid and write sub-sum to shared memory
    static __shared__ float warpSums[32];
    if ((tid & 31) == 0) warpSums[tid / 32] = sum;

    // Make sure all threads have written their values to shared memory
    __syncthreads();

    // Finally sum up the sub-sums of each warp group in the first warp
    int32_t finalSum = (tid < blockDim.x / 32) ? warpSums[tid] : 0;
    if (tid < 32) {
        finalSum = warpReduceSum(finalSum);
        if (tid == 0) out[row] = finalSum;
    }
}


void pretty_print(int32_t size, int32_t *vec_a, int32_t *vec_b, int32_t *mat)
{
    std::cout << "Vec A:" << std::endl;
    for (auto i = 0; i < size; i++)
        std::cout << vec_a[i] << std::endl;

    std::cout << "Vec B:" << std::endl;
    for (auto i = 0; i < size; i++)
        std::cout << vec_b[i] << std::endl;

    std::cout << "Matrix:" << std::endl;
    for (auto i = 0; i < size; i++)
    {
        for (auto j = 0; j < size; j++)
            std::cout << mat[i * size + j] << " ";

        std::cout << std::endl;
    }
}

void check(cudaError_t err, const char *msg) {
    if (err != cudaSuccess)
    {
        std::cerr << "Error: " << msg << " : " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    cudaError_t err = cudaSuccess;
    // int32_t size = 4;
    int32_t size = 32768;
    // If we have less columns than threads, save the remaining ones
    int32_t threads = 1024;
    if (threads > size) {
        // Use a multiple of 32 threads when we have less columns than threads
        threads = (size + 31) & ~31;
    }
    // Use a block for every row
    int32_t blocks = size;
    std::cout << "Blocks: " << blocks << std::endl;
    std::cout << "Threads: " << threads << std::endl << std::endl;

    // CPU MEM
    auto cpu_mem_start = std::chrono::system_clock::now();
    auto vec_a = (int32_t *)malloc(sizeof(int32_t) * size);
    auto vec_b = (int32_t *)malloc(sizeof(int32_t) * size);
    auto mat = (int32_t *)malloc(sizeof(int32_t *) * size * size);
    auto out = (int32_t *)malloc(sizeof(int32_t) * size);
    auto cpu_mem_end = std::chrono::system_clock::now();
    std::chrono::duration<double> cpu_mem_duration = cpu_mem_end - cpu_mem_start;

    init(size, vec_a, vec_b, mat);

    // GPU MEM
    auto gpu_mem_in_start = std::chrono::system_clock::now();
    auto gpu_out = (int32_t *)malloc(sizeof(int32_t) * size);
    int32_t *d_vec_a, *d_vec_b, *d_mat, *d_vec_sum, *d_row_sum, *d_out;
    err = cudaMalloc(&d_vec_a, sizeof(int32_t) * size);
    check(err, "cudaMalloc failed for d_vec_a");

    err = cudaMalloc(&d_vec_b, sizeof(int32_t) * size);
    check(err, "cudaMalloc failed for d_vec_b");

    err = cudaMalloc(&d_mat, sizeof(int32_t) * size * size);
    check(err, "cudaMalloc failed for d_mat");


    err = cudaMalloc(&d_vec_sum, sizeof(int32_t) * size);
    check(err, "cudaMalloc failed for d_vec_sum");

    err = cudaMalloc(&d_row_sum, sizeof(int32_t) * size);
    check(err, "cudaMalloc failed for d_row_sum");

    err = cudaMalloc(&d_out, sizeof(int32_t) * size);
    check(err, "cudaMalloc failed for d_out");

    err = cudaMemcpy(d_vec_a, vec_a, sizeof(int32_t) * size, cudaMemcpyHostToDevice);
    check(err, "cudaMemcpy failed for d_vec_a");

    err = cudaMemcpy(d_vec_b, vec_b, sizeof(int32_t) * size, cudaMemcpyHostToDevice);
    check(err, "cudaMemcpy failed for d_vec_b");

    err = cudaMemcpy(d_mat, mat, sizeof(int32_t) * size * size, cudaMemcpyHostToDevice);
    check(err, "cudaMemcpy failed for d_mat");

    auto gpu_mem_in_end = std::chrono::system_clock::now();
    std::chrono::duration<double> gpu_mem_in_duration = gpu_mem_in_end - gpu_mem_in_start;


    auto gpu_compute_start = std::chrono::system_clock::now();
    hadamard<<<blocks, threads>>>(size, d_vec_a, d_vec_b, d_mat, d_out);
    cudaDeviceSynchronize();
    auto gpu_compute_end = std::chrono::system_clock::now();
    std::chrono::duration<double> gpu_compute_duration = gpu_compute_end - gpu_compute_start;

    auto gpu_mem_out_start = std::chrono::system_clock::now();
    err = cudaMemcpy(gpu_out, d_out, sizeof(int32_t) * size, cudaMemcpyDeviceToHost);
    check(err, "cudaMemcpy failed for gpu_out");
    auto gpu_mem_out_end = std::chrono::system_clock::now();
    std::chrono::duration<double> gpu_mem_out_duration = gpu_mem_out_end - gpu_mem_out_start;
    std::chrono::duration<double> gpu_mem_duration = gpu_mem_in_duration + gpu_mem_out_duration;
    std::chrono::duration<double> gpu_duration = gpu_mem_duration + gpu_compute_duration;

    std::cout << "GPU MEM: elapsed time: " << gpu_mem_duration.count() << "s" << std::endl;
    std::cout << "GPU COMPUTE: elapsed time: " << gpu_compute_duration.count() << "s" << std::endl;
    std::cout << "GPU: elapsed time: " << gpu_duration.count() << "s" << std::endl << std::endl;

    std::cout << "GPU: First 3 entries of Out Vec:" << std::endl;
    for (int32_t i = 0; i < 3; i++)
        std::cout << gpu_out[i] << std::endl;
    std::cout << std::endl;


    auto cpu_compute_start = std::chrono::system_clock::now();
    compute(size, vec_a, vec_b, mat, out);
    auto cpu_compute_end = std::chrono::system_clock::now();
    std::chrono::duration<double> cpu_compute_duration = cpu_compute_end - cpu_compute_start;
    std::chrono::duration<double> cpu_duration = cpu_mem_duration + cpu_compute_duration;

    std::cout << "CPU MEM: elapsed time: " << cpu_mem_duration.count() << "s" << std::endl;
    std::cout << "CPU COMPUTE: elapsed time: " << cpu_compute_duration.count() << "s" << std::endl;
    std::cout << "CPU: elapsed time: " << cpu_duration.count() << "s" << std::endl << std::endl;

    std::cout << "CPU: First 3 entries of Out Vec:" << std::endl;
        for (int32_t i = 0; i < 3; i++)
            std::cout << out[i] << std::endl;
    std::cout << std::endl;

    free(vec_a);
    free(vec_b);
    free(mat);
    free(out);
    free(gpu_out);

    err = cudaFree(d_vec_a);
    check(err, "cudaFree failed for d_vec_a");
    err = cudaFree(d_vec_b);
    check(err, "cudaFree failed for d_vec_b");
    err = cudaFree(d_mat);
    check(err, "cudaFree failed for d_mat");
    err = cudaFree(d_vec_sum);
    check(err, "cudaFree failed for d_vec_sum");
    err = cudaFree(d_row_sum);
    check(err, "cudaFree failed for d_row_sum");
    err = cudaFree(d_out);
    check(err, "cudaFree failed for d_out");

    return 0;
}
