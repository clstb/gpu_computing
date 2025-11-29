#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  if ((call) != cudaSuccess) {                                                 \
    std::cerr << "CUDA error at " << __LINE__ << std::endl;                    \
    exit(EXIT_FAILURE);                                                        \
  }

const int NUM_MATRICES = 10;
const int MATRIX_SIZE = 4096;
const int TILE_SIZE = 32;

__global__ void matrixMultiplyKernel(const float *A, const float *B, float *C,
                                     int n) {
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int col = threadIdx.x + blockIdx.x * blockDim.x;

  if (row < n && col < n) {
    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
      sum += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = sum;
  }
}

__global__ void matrixMultiplyKernelTiled(const float *A, const float *B,
                                          float *C, int n) {
  __shared__ float tileA[TILE_SIZE][TILE_SIZE];
  __shared__ float tileB[TILE_SIZE][TILE_SIZE];
  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  float sum = 0.0f;

  for (int m = 0; m < n; m += TILE_SIZE) {
    if (row < n && (m + threadIdx.x) < n)
      tileA[threadIdx.y][threadIdx.x] = A[row * n + (m + threadIdx.x)];
    else
      tileA[threadIdx.y][threadIdx.x] = 0.0f;

    if (col < n && (m + threadIdx.y) < n)
      tileB[threadIdx.y][threadIdx.x] = B[(m + threadIdx.y) * n + col];
    else
      tileB[threadIdx.y][threadIdx.x] = 0.0f;
    __syncthreads();

    for (int k = 0; k < TILE_SIZE; ++k) {
      sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < n && col < n) {
    C[row * n + col] = sum;
  }
}

void matrixMultiplyNoStreams() {
  float *h_A[NUM_MATRICES], *h_B[NUM_MATRICES], *h_C[NUM_MATRICES];
  float *d_A[NUM_MATRICES], *d_B[NUM_MATRICES], *d_C[NUM_MATRICES];

  for (int i = 0; i < NUM_MATRICES; i++) {
    h_A[i] = (float *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    h_B[i] = (float *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    h_C[i] = (float *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    for (int j = 0; j < MATRIX_SIZE * MATRIX_SIZE; j++) {
      h_A[i][j] = 1.0f;
      h_B[i][j] = 0.01f;
      h_C[i][j] = 0.0f;
    }

    CHECK_CUDA(cudaMalloc(&d_A[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A[i], h_A[i],
                          MATRIX_SIZE * MATRIX_SIZE * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B[i], h_B[i],
                          MATRIX_SIZE * MATRIX_SIZE * sizeof(float),
                          cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(MATRIX_SIZE / TILE_SIZE, MATRIX_SIZE / TILE_SIZE);

    std::cout << "Launch kernel with " << blocksPerGrid.x * blocksPerGrid.y
              << " blocks each with " << threadsPerBlock.x * threadsPerBlock.y
              << " threads\n";
    // matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A[i], d_B[i],
    // d_C[i], MATRIX_SIZE);
    matrixMultiplyKernelTiled<<<blocksPerGrid, threadsPerBlock>>>(
        d_A[i], d_B[i], d_C[i], MATRIX_SIZE);
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpy(h_C[i], d_C[i],
                          MATRIX_SIZE * MATRIX_SIZE * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // double eps = 1.e-6;
    // for (int j = 0; j < MATRIX_SIZE * MATRIX_SIZE; j++) {
    //   double abs_err = fabs(h_C[i][j] - (MATRIX_SIZE * 0.01f));
    //   double dot_length = MATRIX_SIZE;
    //   double abs_val = fabs(h_C[i][j]);
    //   double rel_err = abs_err / abs_val / dot_length;

    //   if (rel_err > eps) {
    //     printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", j,
    //            h_C[i][j], MATRIX_SIZE * 0.01f, eps);
    //   }
    // }

    free(h_A[i]);
    free(h_B[i]);
    free(h_C[i]);
    cudaFree(d_A[i]);
    cudaFree(d_B[i]);
    cudaFree(d_C[i]);
  }
}

void matrixMultiplyWithStreams() {
  float *h_A[NUM_MATRICES], *h_B[NUM_MATRICES], *h_C[NUM_MATRICES];
  float *d_A[NUM_MATRICES], *d_B[NUM_MATRICES], *d_C[NUM_MATRICES];

  cudaStream_t streams[NUM_MATRICES];

  for (int i = 0; i < NUM_MATRICES; i++) {
    cudaMallocHost((void **)&h_A[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMallocHost((void **)&h_B[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMallocHost((void **)&h_C[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    for (int j = 0; j < MATRIX_SIZE * MATRIX_SIZE; j++) {
      h_A[i][j] = 1.0f;
      h_B[i][j] = 0.01f;
      h_C[i][j] = 0.0f;
    }

    cudaMalloc((void **)&d_A[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&d_B[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **)&d_C[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    cudaStreamCreate(&streams[i]);

    cudaMemcpyAsync(d_A[i], h_A[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float),
                    cudaMemcpyHostToDevice, streams[i]);
    cudaMemcpyAsync(d_B[i], h_B[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float),
                    cudaMemcpyHostToDevice, streams[i]);
  }

  for (int i = 0; i < NUM_MATRICES; i++) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(MATRIX_SIZE / TILE_SIZE, MATRIX_SIZE / TILE_SIZE);

    std::cout << "Launch kernel with " << blocksPerGrid.x * blocksPerGrid.y
              << " blocks each with " << threadsPerBlock.x * threadsPerBlock.y
              << " threads"
              << " stream " << i << std::endl;

    matrixMultiplyKernelTiled<<<blocksPerGrid, threadsPerBlock, 0,
                                streams[i]>>>(d_A[i], d_B[i], d_C[i],
                                              MATRIX_SIZE);
  }

  for (int i = 0; i < NUM_MATRICES; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }
  cudaDeviceSynchronize();

  for (int i = 0; i < NUM_MATRICES; i++) {
    cudaMemcpyAsync(h_C[i], d_C[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float),
                    cudaMemcpyDeviceToHost, streams[i]);
  }

  // double eps = 1.e-6; // machine zero
  // for (int i = 1; i < NUM_MATRICES; i++) {
  //   for (int j = 0; j < MATRIX_SIZE * MATRIX_SIZE; j++) {
  //     double abs_err = fabs(h_C[i][j] - (MATRIX_SIZE * 0.01f));
  //     double dot_length = MATRIX_SIZE;
  //     double abs_val = fabs(h_C[i][j]);
  //     double rel_err = abs_err / abs_val / dot_length;

  //     if (rel_err > eps) {
  //       printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", j,
  //              h_C[i][j], MATRIX_SIZE * 0.01f, eps);
  //     }
  //   }
  // }

  for (int i = 0; i < NUM_MATRICES; i++) {
    cudaFree(d_A[i]);
    cudaFree(d_B[i]);
    cudaFree(d_C[i]);

    cudaFreeHost(h_A[i]);
    cudaFreeHost(h_B[i]);
    cudaFreeHost(h_C[i]);
  }
}

int main() {
  matrixMultiplyWithStreams();
  matrixMultiplyNoStreams();
  return EXIT_SUCCESS;
}
