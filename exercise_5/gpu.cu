#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

#define N 128                // Grid size X
#define M 128                // Grid size Y
#define ITERATIONS 100000    // Number of iterations
#define DIFFUSION_FACTOR 0.5 // Diffusion factor
#define CELL_SIZE 0.01       // Cell size for the simulation
#define BLOCK_SIZE 16        // CUDA block size

void initializeGrid(float *grid, int n, int m)
{
    for (int y = 0; y < m; ++y)
    {
        for (int x = 0; x < n; ++x)
        {
            // Initialize one quadrant to a high temp
            // and the rest to 0.
            if (y > m / 2 && x > n / 2)
            {
                grid[y * n + x] = 100.0f; // Temp in corner
            }
            else
            {
                grid[y * n + x] = 0.0f; // Temp in the rest
            }
        }
    }
}

// heat  simulation on gpu
__global__ void heatKernel(float *curr, float *next, int n, int m, float dt)
{
    __shared__ float temp[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    int block_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int block_y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    int halo_x = threadIdx.x + 1;
    int halo_y = threadIdx.y + 1;

    // load data
    // center
    if (block_x < n && block_y < m)
    {
        temp[halo_y][halo_x] = curr[block_y * n + block_x];
    }

    // top
    if (threadIdx.y == 0 && block_y > 0)
    {
        temp[0][halo_x] = curr[(block_y - 1) * n + block_x];
    }
    // bottom
    if (threadIdx.y == BLOCK_SIZE - 1 && block_y < m - 1)
    {
        temp[BLOCK_SIZE + 1][halo_x] = curr[(block_y + 1) * n + block_x];
    }
    // left
    if (threadIdx.x == 0 && block_x > 0)
    {
        temp[halo_y][0] = curr[block_y * n + block_x - 1];
    }
    // right
    if (threadIdx.x == BLOCK_SIZE - 1 && block_x < n - 1)
    {
        temp[halo_y][BLOCK_SIZE + 1] = curr[block_y * n + block_x + 1];
    }

    __syncthreads();

    // compute, similar to cpu
    if (block_x > 0 && block_x < n - 1 && block_y > 0 && block_y < m - 1)
    {
        float dx2 = CELL_SIZE * CELL_SIZE;
        float dy2 = CELL_SIZE * CELL_SIZE;

        auto left = temp[halo_y][halo_x - 1];
        auto right = temp[halo_y][halo_x + 1];
        auto below = temp[halo_y - 1][halo_x];
        auto above = temp[halo_y + 1][halo_x];
        auto center = temp[halo_y][halo_x];

        next[block_y * n + block_x] = center + DIFFUSION_FACTOR * dt *
                                                   ((left - 2.0 * center + right) / dy2 +
                                                    (above - 2.0 * center + below) / dx2);
    }
}

// orchestrator for heat simulation on gpu
float *heatSimulation(float *curr, float *next, int n, int m, int iterations, float dt)
{
    float *d_curr, *d_next;
    size_t size = n * m * sizeof(float);

    // allocate memory
    cudaMalloc(&d_curr, size);
    cudaMalloc(&d_next, size);

    // copy data
    cudaMemcpy(d_curr, curr, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_next, next, size, cudaMemcpyHostToDevice);

    // grid sizing
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // call kernel iteratively
    for (int iter = 0; iter < iterations; ++iter)
    {
        heatKernel<<<numBlocks, threadsPerBlock>>>(d_curr, d_next, n, m, dt);
        std::swap(d_curr, d_next);
    }

    // copy result back
    cudaMemcpy(curr, d_curr, size, cudaMemcpyDeviceToHost);

    // free
    cudaFree(d_curr);
    cudaFree(d_next);

    return curr;
}

int main()
{
    // Allocate memory for the grids
    float *curr = (float *)malloc(N * M * sizeof(float));
    float *next = (float *)malloc(N * M * sizeof(float));

    // Check for allocation failures
    if (curr == NULL || next == NULL)
    {
        std::cerr << "Memory allocation failed!" << std::endl;
        return EXIT_FAILURE;
    }

    // Initialize the grids
    initializeGrid(curr, N, M);
    initializeGrid(next, N, M);

    float dx2 = CELL_SIZE * CELL_SIZE;
    float dy2 = CELL_SIZE * CELL_SIZE;
    float dt = dx2 * dy2 / (2.0 * DIFFUSION_FACTOR * (dx2 + dy2));

    // Run the heat simulation
    auto final_grid = heatSimulation(curr, next, N, M, ITERATIONS, dt);

    // Print a small section of the final grid for verification
    std::cout << "Final grid values (top-left corner):" << std::endl;
    for (int y = 0; y < 16; ++y)
    {
        for (int x = 0; x < 16; ++x)
        {
            std::cout << std::setw(6) << std::fixed << std::setprecision(2) << final_grid[y * N + x] << " ";
        }
        std::cout << std::endl;
    }

    // Free allocated memory
    free(curr);
    free(next);

    return 0;
}
