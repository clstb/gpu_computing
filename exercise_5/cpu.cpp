#include <iostream>
#include <iomanip>

#define N 128                // Grid size X
#define M 128                // Grid size Y
#define ITERATIONS 100000    // Number of iterations
#define DIFFUSION_FACTOR 0.5 // Diffusion factor
#define CELL_SIZE 0.01       // Cell size for the simulation

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

// Function to run the heat simulation on the CPU
float *heatSimulation(float *curr, float *next, int n, int m, int iterations, float dt)
{
    float dx2 = CELL_SIZE * CELL_SIZE;
    float dy2 = CELL_SIZE * CELL_SIZE;

    for (int iter = 0; iter < iterations; ++iter)
    {
        for (int y = 1; y < m - 1; ++y)
        {
            for (int x = 1; x < n - 1; ++x)
            {
                auto left = curr[y * m + x - 1];
                auto right = curr[y * m + x + 1];
                auto below = curr[(y - 1) * m + x];
                auto above = curr[(y + 1) * m + x];
                auto center = curr[y * m + x];
                next[y * m + x] = center + DIFFUSION_FACTOR * dt *
                                               ((left - 2.0 * center + right) / dy2 +
                                                (above - 2.0 * center + below) / dx2);
            }
        }
        // Swap the current and next grids
        std::swap(curr, next);
    }
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
