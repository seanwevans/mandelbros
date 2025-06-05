
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

#define BLOCK_SIZE 128
#define MAX_ITERATIONS 1000

__global__ void mandelbrotKernel(double *x, double *y, int *iterations,
                                  int width, int height)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height)
        return;

    double real = x[i];
    double imag = y[j];
    double real2 = real * real;
    double imag2 = imag * imag;
    int count = 0;

    while (count < MAX_ITERATIONS && real2 + imag2 <= 4.0)
    {
        imag = 2.0 * real * imag + y[j];
        real = real2 - imag2 + x[i];

        real2 = real * real;
        imag2 = imag * imag;
        count++;
    }

    iterations[j * width + i] = count;
}

int main()
{
    int width = 800, height = 800;
    double xmin = -2.0, xmax = 1.0;
    double ymin = -1.5, ymax = 1.5;
    double stepx = (xmax - xmin) / width;
    double stepy = (ymax - ymin) / height;

    double *x, *y;
    int *iterations;
    cudaMallocManaged(&x, width * sizeof(double));
    cudaMallocManaged(&y, height * sizeof(double));
    cudaMallocManaged(&iterations, width * height * sizeof(int));

    for (int i = 0; i < width; i++)
        x[i] = xmin + i * stepx;

    for (int j = 0; j < height; j++)
        y[j] = ymin + j * stepy;

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    mandelbrotKernel<<<grid, block>>>(x, y, iterations, width, height);

    cudaDeviceSynchronize();

    for (int i = 0; i < width * height; i++)
        std::cout << iterations[i] << " ";

    cudaFree(x);
    cudaFree(y);
    cudaFree(iterations);

    return 0;
}
