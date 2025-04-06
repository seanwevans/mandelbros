#include <gmp.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

#define BLOCK_SIZE 128
#define MAX_ITERATIONS 1000

__global__ void mandelbrotKernel(mpf_t *x, mpf_t *y, int *iterations, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx % width;
    int j = idx / width;

    mpf_t real, imag, temp, real2, imag2;
    mpf_init(real);
    mpf_init(imag);
    mpf_init(temp);
    mpf_init(real2);
    mpf_init(imag2);

    mpf_set(real, x[i]);
    mpf_set(imag, y[j]);

    int count = 0;
    while (count < MAX_ITERATIONS)
    {
        mpf_mul(real2, real, real);
        mpf_mul(imag2, imag, imag);

        if (mpf_cmp(real2 + imag2, (mpf_t)2.0) > 0)
            break;

        mpf_mul(temp, real, imag);
        mpf_add(imag, temp, temp);
        mpf_add(imag, imag, y[j]);

        mpf_set(real, real2 - imag2 + x[i]);

        count++;
    }

    iterations[idx] = count;

    mpf_clear(real);
    mpf_clear(imag);
    mpf_clear(temp);
    mpf_clear(real2);
    mpf_clear(imag2);
}

int main()
{
    int width = 800, height = 800;
    mpf_t xmin, xmax, ymin, ymax, step;
    mpf_init(xmin);
    mpf_init(xmax);
    mpf_init(ymin);
    mpf_init(ymax);
    mpf_init(step);

    mpf_set_d(xmin, -2.0);
    mpf_set_d(xmax, 1.0);
    mpf_set_d(ymin, -1.5);
    mpf_set_d(ymax, 1.5);
    mpf_sub(step, xmax, xmin);
    mpf_div_ui(step, step, width);

    mpf_t *x, *y;
    int *iterations;
    cudaMalloc(&x, width * sizeof(mpf_t));
    cudaMalloc(&y, height * sizeof(mpf_t));
    cudaMalloc(&iterations, width * height * sizeof(int));

    for (int i = 0; i < width; i++)
    {
        mpf_init(x[i]);
        mpf_mul_ui(x[i], step, i);
        mpf_add(x[i], x[i], xmin);
    }

    for (int j = 0; j < height; j++)
    {
        mpf_init(y[j]);
        mpf_mul_ui(y[j], step, j);
        mpf_add(y[j], y[j], ymin);
    }

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    mandelbrotKernel<<<(x, y, iterations, width, height)

    cudaDeviceSynchronize();

    for (int i = 0; i < width * height; i++)
        std::cout << iterations[i] << " ";

    cudaFree(x);
    cudaFree(y);
    cudaFree(iterations);

    return 0;
}
