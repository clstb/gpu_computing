// Build: nvcc -arch sm_75 -o task task.cu

#include <iostream>
#include <mma.h>
#include <random>
#include <utility>

using namespace nvcuda;

#define NUM_KERNELS 16 // How many kernels we have to convolve against the input
#define KERNEL_SIZE                                                            \
  16 // The length of each kernel vector, and equivalently the size of each
     // window
#define INPUT_SIZE (32000 - 1) // The length of the input vector.
#define OUTPUT_SIZE                                                            \
  (INPUT_SIZE - KERNEL_SIZE + 1) // The length of the output vector, or
                                 // equivalently the number of windows.
#define CHUNK_SIZE                                                             \
  16 // we group CHUNK_SIZE-many consecutive windows into a chunk.
#define NUM_CHUNKS                                                             \
  (OUTPUT_SIZE / CHUNK_SIZE) // How many chunks we have to compute.

// Write A[idx(i, j, N, K)] to index A_i,j into an NxK matrix.
#define idx(i, j, N, K) (i * K + j)

// Simplifying assumption: the produced output size is divisible by our chunk
// size.
static_assert(OUTPUT_SIZE % CHUNK_SIZE == 0);

#define CUDA_CALL(x)                                                           \
  do {                                                                         \
    cudaError_t error = x;                                                     \
    if (error != cudaSuccess) {                                                \
      const char *cuda_err_str = cudaGetErrorString(error);                    \
      std::cerr << "Cuda Error at" << __FILE__ << ":" << __LINE__ << ": "      \
                << cuda_err_str << std::endl;                                  \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

// Produces a random vector, and copies it over to the GPU.
std::pair<half *, half *> random_vec(size_t n) {
  static std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  half *cpu_out = (half *)malloc(sizeof(half) * n);
  for (int i = 0; i < n; i++) {
    cpu_out[i] = (half)dist(rng);
  }

  half *gpu_out;
  cudaError_t error;
  error = cudaMalloc(&gpu_out, sizeof(half) * n);
  if (error != cudaSuccess) {
    std::cerr << "Failed to cudaMalloc in random_vec"
              << cudaGetErrorString(error) << std::endl;
    return {nullptr, nullptr};
  }
  error =
      cudaMemcpy(gpu_out, cpu_out, sizeof(half) * n, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    std::cerr << "Failed to cudaMemcpy in random_vec"
              << cudaGetErrorString(error) << std::endl;
    return {nullptr, nullptr};
  }
  return {cpu_out, gpu_out};
}

bool almost_equal(float a, float b, float eps = 1e-2f) {
  float diff = fabsf(a - b);
  return diff <= eps * fmaxf(1.0f, fmaxf(fabsf(a), fabsf(b)));
}

///////////////////////////////////////////////////////////////
// CPU 1 - A naive convolution
///////////////////////////////////////////////////////////////
void convolve_cpu(half *out, half *input, half *kernels) {
  for (int p = 0; p < OUTPUT_SIZE; p++) {
    for (int k = 0; k < NUM_KERNELS; k++) {
      half acc = 0.0;
      for (int o = 0; o < KERNEL_SIZE; o++) {
        acc += kernels[idx(o, k, KERNEL_SIZE, NUM_KERNELS)] * input[p + o];
      }
      out[idx(p, k, OUTPUT_SIZE, NUM_KERNELS)] = acc;
    }
  }
}

///////////////////////////////////////////////////////////////
// CPU 2 - A convolution using matrix multiplication
///////////////////////////////////////////////////////////////

void matmul(half *a, half *b, half *c) {
  // a :: N x K
  // b :: K x M
  // c :: N x M

  // We fix N, M, K, as tensor cores work with fixed size matrices.
  const int N = 16;
  const int M = 16;
  const int K = 16;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      half acc = 0.0;
      for (int k = 0; k < K; k++) {
        acc += a[idx(i, k, N, K)] * b[idx(k, j, K, M)];
      }
      c[idx(i, j, N, M)] = acc;
    }
  }
}

// already closer to a GPU implementation.
void convolve_cpu2(half *out, half *input, half *kernels) {
  for (int s = 0; s < NUM_CHUNKS; s++) {
    half a[CHUNK_SIZE * KERNEL_SIZE];
    half b[KERNEL_SIZE * NUM_KERNELS];
    half c[CHUNK_SIZE * NUM_KERNELS];

    // init A:
    for (int w = 0; w < CHUNK_SIZE; w++) {
      auto p = s * CHUNK_SIZE + w;
      for (int o = 0; o < KERNEL_SIZE; o++) {
        a[idx(w, o, CHUNK_SIZE, KERNEL_SIZE)] = input[p + o];
      }
    }

    // init B:
    for (int k = 0; k < NUM_KERNELS; k++) {
      for (int o = 0; o < KERNEL_SIZE; o++) {
        b[idx(o, k, KERNEL_SIZE, NUM_KERNELS)] =
            kernels[idx(o, k, KERNEL_SIZE, NUM_KERNELS)];
      }
    }

    matmul(a, b, c);

    for (int k = 0; k < NUM_KERNELS; k++) {
      for (int w = 0; w < CHUNK_SIZE; w++) {
        auto p = s * CHUNK_SIZE + w;
        out[idx(p, k, OUTPUT_SIZE, NUM_KERNELS)] =
            c[idx(w, k, CHUNK_SIZE, NUM_KERNELS)];
      }
    }
  }
}

/////////////////////
// GPU
/////////////////////
__global__ void convolve_gpu(half *out, half *input, half *kernels) {
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b;
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c;

  wmma::fill_fragment(frag_c, 0.0f);

  int chunk_idx = blockIdx.x;
  int chunk_start_input = chunk_idx * CHUNK_SIZE;
  int tid = threadIdx.x;

  __shared__ half mem[CHUNK_SIZE * KERNEL_SIZE]; // 16 * 16

  for (int i = 0; i < 8; i++) {
    int offset = tid + i * 32;
    int row = offset / KERNEL_SIZE;
    int col = offset % KERNEL_SIZE;

    if (offset < CHUNK_SIZE * KERNEL_SIZE) {
      mem[offset] = input[chunk_start_input + row + col];
    }
  }

  __syncthreads();

  wmma::load_matrix_sync(frag_a, mem, KERNEL_SIZE);
  wmma::load_matrix_sync(frag_b, kernels, NUM_KERNELS);
  wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
  half *out_ptr = &out[chunk_idx * CHUNK_SIZE * NUM_KERNELS];
  wmma::store_matrix_sync(out_ptr, frag_c, NUM_KERNELS, wmma::mem_row_major);
}

int main() {
  ///////////////////
  // INIT
  ///////////////////

  // input :: INPUT_SIZE
  auto [cpu_input, gpu_input] = random_vec(INPUT_SIZE);

  // kernels :: KERNEL_SIZE x NUM_KERNELS
  auto [cpu_kernels, gpu_kernels] = random_vec(KERNEL_SIZE * NUM_KERNELS);

  // out :: OUTPUT_SIZE x NUM_KERNELS
  auto [cpu_out, gpu_out] = random_vec(OUTPUT_SIZE * NUM_KERNELS);
  half *cpu_out2 = (half *)malloc(sizeof(half) * OUTPUT_SIZE * OUTPUT_SIZE);

  ///////////////////
  // CALL CPU
  ///////////////////
  convolve_cpu(cpu_out, cpu_input, cpu_kernels);
  convolve_cpu2(cpu_out2, cpu_input, cpu_kernels);

  ///////////////////
  // CALL GPU
  ///////////////////
  convolve_gpu<<<NUM_CHUNKS, 32>>>(gpu_out, gpu_input, gpu_kernels);
  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());

  ///////////////////
  // COMPARE
  ///////////////////
  half *cpu_out_from_gpu =
      (half *)malloc(sizeof(half) * OUTPUT_SIZE * NUM_KERNELS);
  CUDA_CALL(cudaMemcpy(cpu_out_from_gpu, gpu_out,
                       sizeof(half) * OUTPUT_SIZE * NUM_KERNELS,
                       cudaMemcpyDeviceToHost));

  for (int k = 0; k < NUM_KERNELS; k++) {
    for (int p = 0; p < OUTPUT_SIZE; p++) {
      float cpu1 = __half2float(cpu_out[idx(p, k, OUTPUT_SIZE, NUM_KERNELS)]);
      float cpu2 = __half2float(cpu_out2[idx(p, k, OUTPUT_SIZE, NUM_KERNELS)]);
      float gpu =
          __half2float(cpu_out_from_gpu[idx(p, k, OUTPUT_SIZE, NUM_KERNELS)]);

      if (!almost_equal(cpu1, cpu2) || !almost_equal(cpu1, gpu)) {
        std::cout << "INCORRECT: " << cpu1 << " ~ " << cpu2 << " ~ " << gpu
                  << std::endl;
      }
    }
  }

  free(cpu_input);
  free(cpu_kernels);
  free(cpu_out);
  free(cpu_out2);
  free(cpu_out_from_gpu);
  CUDA_CALL(cudaFree(gpu_kernels));
  CUDA_CALL(cudaFree(gpu_input));
  CUDA_CALL(cudaFree(gpu_out));
}
