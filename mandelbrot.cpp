#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

#define BLOCK_SIZE 128
#define MAX_ITERATIONS 1000

__global__ void mandelbrotKernel(double *x, double *y, int *iterations, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx % width;
    int j = idx / width;

    double real = x[i];
    double imag = y[j];

    int count = 0;
    while (count < MAX_ITERATIONS)
    {
        double real2 = real * real;
        double imag2 = imag * imag;

        if (real2 + imag2 > 4.0)
            break;

        double temp = real * imag;
        imag = 2.0 * temp + y[j];
        real = real2 - imag2 + x[i];

        count++;
    }

    iterations[idx] = count;
}

int main()
{
    int width = 800, height = 800;
    double xmin = -2.0;
    double xmax = 1.0;
    double ymin = -1.5;
    double ymax = 1.5;
    double step = (xmax - xmin) / width;

    double *hx = new double[width];
    double *hy = new double[height];
    int *hIterations = new int[width * height];

    for (int i = 0; i < width; i++)
        hx[i] = xmin + step * i;

    for (int j = 0; j < height; j++)
        hy[j] = ymin + step * j;

    double *x, *y;
    int *iterations;
    cudaMalloc(&x, width * sizeof(double));
    cudaMalloc(&y, height * sizeof(double));
    cudaMalloc(&iterations, width * height * sizeof(int));

    cudaMemcpy(x, hx, width * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y, hy, height * sizeof(double), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid((width * height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    mandelbrotKernel<<<grid, block>>>(x, y, iterations, width, height);

    cudaDeviceSynchronize();

    cudaMemcpy(hIterations, iterations, width * height * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < width * height; i++)
        std::cout << hIterations[i] << " ";

    cudaFree(x);
    cudaFree(y);
    cudaFree(iterations);

    delete[] hx;
    delete[] hy;
    delete[] hIterations;

    return 0;
}
