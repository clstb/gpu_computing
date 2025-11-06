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

int main()
{
    // int32_t size = 3;
    int32_t size = 32768;

    auto vec_a = (int32_t *)malloc(sizeof(int32_t) * size);
    auto vec_b = (int32_t *)malloc(sizeof(int32_t) * size);
    // Flat Buffer for matrix
    auto mat = (int32_t *)malloc(sizeof(int32_t *) * size * size);
    auto out = (int32_t *)malloc(sizeof(int32_t) * size);

    init(size, vec_a, vec_b, mat);

    // pretty_print(size, vec_a, vec_b, mat);

    auto start = std::chrono::system_clock::now();
    compute(size, vec_a, vec_b, mat, out);
    auto end = std::chrono::system_clock::now();

    std::cout << "First 3 entries of Out Vec:" << std::endl;
    for (int32_t i = 0; i < 3; i++)
        std::cout << out[i] << std::endl;

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

    free(vec_a);
    free(vec_b);
    free(mat);
    free(out);

    return 0;
}
