#include <chrono>
#include <curand.h>
#include <iostream>
#include <stdlib.h>

#include "helper.cu"

unsigned const int BLOCK_SIZE = 1024;
__device__ const unsigned int WARP_SIZE = 32;
__device__ const unsigned FULL_MASK = 0xFFFFFFFF; // All threads of the warp participate in the shuffle
__device__ unsigned int block_counter = 0;

// Dynamic block ID generation
// This is required for block-level streaming synchronization
// as block order is not guaranteed by the hardware scheduler
__inline__ __device__ unsigned int getBlockId() {
    __shared__ unsigned int s_bid;
    if (threadIdx.x == 0) {
        s_bid = atomicAdd(&block_counter, 1);
    }
    __syncthreads();
    return s_bid;
}

__inline__ __device__ bool isLastThreadInWarp(int thread_id, int warp_id) {
    const int thread_warp_id = thread_id & (WARP_SIZE - 1);
    if (thread_warp_id == WARP_SIZE - 1) return true;
    const int warp_start = warp_id * WARP_SIZE;
    const int warp_end = min(warp_start + WARP_SIZE - 1, BLOCK_SIZE - 1);
    return thread_id == warp_end;
}


__inline__ __device__ void complexMul(int &thread_warp_id, float &re, float &im) {
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        // We can only shuffle single values, so we need one for each part of the complex number
        float n_re = __shfl_up_sync(FULL_MASK, re, offset);
        float n_im = __shfl_up_sync(FULL_MASK, im, offset);

        if (thread_warp_id >= offset) {
            float tmp_re = n_re * re - n_im * im;
            float tmp_im = n_re * im + re * n_im;
            re = tmp_re;
            im = tmp_im;
        }
    }
}

__global__ void koggeStoneStreamingComplexMul(
    int size,
    int *flags,
    float *carries_re,
    float *carries_im,
    float *in_re,
    float *in_im,
    float *out_re,
    float *out_im
) {
    int bid = getBlockId();
    int thread_id = threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int thread_warp_id = thread_id % WARP_SIZE;
    int thread_global_id = bid * blockDim.x + thread_id;
    int warps = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;

    // Let's just assume the WARP_SIZE is a power of 2
    int size_upper_bound = (size + WARP_SIZE - 1) & ~(WARP_SIZE - 1);
    if (thread_global_id >= size_upper_bound) {
        return;
    }

    if (bid == 0 && thread_id == 0) {
        atomicExch(&flags[bid], 1);
        atomicExch(&carries_re[bid], 1.0f);
        atomicExch(&carries_im[bid], 0.0f);
    }
    __syncthreads();

    float re = 1.0f;
    float im = 0.0f;

    if (thread_global_id < size) {
        re = in_re[thread_global_id];
        im = in_im[thread_global_id];
    }
    complexMul(thread_warp_id, re, im);

    // Check if it's the last thread of a warp
    // and write the warps result to shared memory
    // We might waste a bit when warps of a block are not fully filled
    // but we don't need to do dynamic allocs this way
    __shared__ float warp_re_acc[WARP_SIZE];
    __shared__ float warp_im_acc[WARP_SIZE];
    if (isLastThreadInWarp(thread_id, warp_id)) {
        warp_re_acc[warp_id] = re;
        warp_im_acc[warp_id] = im;
    }
    __syncthreads();

    __shared__ float carry_re;
    __shared__ float carry_im;
    if (thread_id < WARP_SIZE) {
        float warp_re = warp_re_acc[thread_id];
        float warp_im = warp_im_acc[thread_id];
        complexMul(thread_id, warp_re, warp_im);

        // Compute offset we have to multiply to each warp
        float offset_re = __shfl_up_sync(0xffffffff, warp_re, 1);
        float offset_im = __shfl_up_sync(0xffffffff, warp_im, 1);

        if (thread_id == warps - 1) {
                // Wait until the previous block is done
                while(atomicAdd(&flags[bid], 0) == 0) {}
                // Calculate value to propagate to next block
                carry_re = carries_re[bid];
                carry_im = carries_im[bid];
                carries_re[bid+1] = carry_re * warp_re - carry_im * warp_im;
                carries_im[bid+1] = carry_re * warp_im + carry_im * warp_re;
                __threadfence();
                atomicAdd(&flags[bid+1], 1);
        }

        warp_re_acc[thread_id] = offset_re;
        warp_im_acc[thread_id] = offset_im;
    }
    __syncthreads();

    // Multiply offsets with carry to final offset
    float offset_re = carry_re * warp_re_acc[warp_id] - carry_im * warp_im_acc[warp_id];
    float offset_im = carry_re * warp_im_acc[warp_id] + carry_im * warp_re_acc[warp_id];
    if (warp_id == 0) {
        offset_re = carry_re;
        offset_im = carry_im;
    }

    out_re[thread_global_id] = offset_re * re - offset_im * im;
    out_im[thread_global_id] = offset_re * im + offset_im * re;
}

void sequential_scan(size_t size, float *in_re_h, float *in_im_h, float *out_re_h, float *out_im_h) {
  out_re_h[0] = in_re_h[0];
  out_im_h[0] = in_im_h[0];
  for (auto i = 1; i < size; i++) {
    float real_cur = in_re_h[i];
    float im_cur = in_im_h[i];
    float real_prev = out_re_h[i - 1];
    float im_prev = out_im_h[i - 1];

    out_re_h[i] = real_prev * real_cur - im_prev * im_cur;
    out_im_h[i] = real_prev * im_cur + real_cur * im_prev;
  }
}

int main() {
  size_t size = 33554432;
  float *in_re_d, *in_im_d, *out_re_d, *out_im_d, *carries_re, *carries_im;
  int *flags;
  float *in_re_h, *in_im_h, *out_re_h, *out_im_h;
  float *gpu_out_re_h, *gpu_out_im_h;

  // Allocate on host
  in_re_h = (float *)calloc(size, sizeof(float));
  in_im_h = (float *)calloc(size, sizeof(float));
  out_re_h = (float *)calloc(size, sizeof(float));
  out_im_h = (float *)calloc(size, sizeof(float));
  gpu_out_re_h = (float *)calloc(size, sizeof(float));
  gpu_out_im_h = (float *)calloc(size, sizeof(float));
  CHECK_ALLOC(in_re_h);
  CHECK_ALLOC(in_im_h);
  CHECK_ALLOC(out_re_h);
  CHECK_ALLOC(out_im_h);
  CHECK_ALLOC(gpu_out_re_h);
  CHECK_ALLOC(gpu_out_im_h);

  // Allocate on device
  auto blocks = size / BLOCK_SIZE;
  auto remainder = size % BLOCK_SIZE;
  if (remainder > 0) {
    blocks++;
  }
  std::cout << "blocks: " << blocks << std::endl;
  CUDA_CALL(cudaMalloc((void **)&in_re_d, size * sizeof(float)));
  CUDA_CALL(cudaMalloc((void **)&in_im_d, size * sizeof(float)));
  CUDA_CALL(cudaMalloc((void **)&out_re_d, size * sizeof(float)));
  CUDA_CALL(cudaMalloc((void **)&out_im_d, size * sizeof(float)));
  // Add 1 as buffer for the last block to write to
  CUDA_CALL(cudaMalloc((void **)&flags, (blocks+1) * sizeof(int)));
  CUDA_CALL(cudaMemset(flags, 0, (blocks+1) * sizeof(int)));
  CUDA_CALL(cudaMalloc((void **)&carries_re, (blocks+1) * sizeof(float)));
  CUDA_CALL(cudaMalloc((void **)&carries_im, (blocks+1) * sizeof(float)));

  // Initialize
  // Note: I have adapted the code to store the real and imaginary part in different arrays
  // This should improve memory coalescing
  int e = random_init(size, in_re_d, in_im_d, in_re_h, in_im_h);
  if (e == EXIT_FAILURE)
    return EXIT_FAILURE;

  auto start = std::chrono::system_clock::now();
  sequential_scan(size, in_re_h, in_im_h, out_re_h, out_im_h);
  auto end = std::chrono::system_clock::now();

  auto gpu_start = std::chrono::system_clock::now();
  koggeStoneStreamingComplexMul<<<blocks,BLOCK_SIZE>>>(
      size,
      flags,
      carries_re,
      carries_im,
      in_re_d,
      in_im_d,
      out_re_d,
      out_im_d
  );
  cudaDeviceSynchronize();
  auto gpu_end = std::chrono::system_clock::now();
  // Copy results back to host
  CUDA_CALL(cudaMemcpy(gpu_out_re_h, out_re_d, size * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(gpu_out_im_h, out_im_d, size * sizeof(float), cudaMemcpyDeviceToHost));

  // Print CPU results
  std::cout << "First 3 entries of In Vec:" << std::endl;
  for (int32_t i = 0; i < 3; i++)
    std::cout << in_re_h[i] << "," << in_im_h[i] << std::endl;

  std::cout << "First 3 entries of Out Vec:" << std::endl;
  for (int32_t i = 0; i < 3; i ++)
    std::cout << out_re_h[i] << " + " << out_im_h[i] << std::endl;

  // Print GPU results
  std::cout << "First 3 entries of GPU Out Vec:" << std::endl;
  for (int32_t i = 0; i < 3; i++)
    std::cout << gpu_out_re_h[i] << " + " << gpu_out_im_h[i] << std::endl;

  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "Elapsed CPU time: " << elapsed_seconds.count() << "s" << std::endl;

  std::chrono::duration<double> gpu_elapsed_seconds = gpu_end - gpu_start;
  std::cout << "Elapsed GPU time: " << gpu_elapsed_seconds.count() << "s" << std::endl;

  CUDA_CALL(cudaFree(in_re_d));
  CUDA_CALL(cudaFree(in_im_d));
  CUDA_CALL(cudaFree(out_re_d));
  CUDA_CALL(cudaFree(out_im_d));
  CUDA_CALL(cudaFree(flags));
  CUDA_CALL(cudaFree(carries_re));
  CUDA_CALL(cudaFree(carries_im));


  free(in_re_h);
  free(in_im_h);
  free(out_re_h);
  free(out_im_h);
  free(gpu_out_re_h);
  free(gpu_out_im_h);

  return EXIT_SUCCESS;
}
