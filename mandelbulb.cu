#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <cmath>

#define WIDTH 1000
#define HEIGHT 1000

#define ITERATIONS 50
#define ZOOM_FACTOR 1.5

// Custom viewpoint parameters
#define EYE_X -0.7
#define EYE_Y 0.0
#define EYE_Z 0.0
#define VIEW_X 0.0
#define VIEW_Y 1.0
#define VIEW_Z 0.0

// Color mapping parameters
#define COLOR_SCALE 20
#define COLOR_OFFSET 1

// Bitmap header structure
typedef struct {
    unsigned short type;
    unsigned int size;
    unsigned short reserved1, reserved2;
    unsigned int offset;
} BMPHeader;

typedef struct {
    unsigned int size;
    int width, height;
    unsigned short planes;
    unsigned short bits;
    unsigned int compression;
    unsigned int imagesize;
    int xresolution, yresolution;
    unsigned int ncolours;
    unsigned int importantcolours;
} BMPInfoHeader;

// GPU kernel to calculate the Mandelbulb
__global__ void MandelbulbKernel(int* image, int width, int height, float eyeX, float eyeY, float eyeZ, float viewX, float viewY, float viewZ) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float zoom = pow(ZOOM_FACTOR, -25.0);
    float fov = M_PI / 2.0;
    float aspect = (float)width / height;

    float angle = tan(fov / 2.0);
    float dx = (2 * (x - width / 2.0) / width) * angle * aspect * zoom;
    float dy = (2 * (y - height / 2.0) / height) * angle * zoom;

    float dirX = dx * viewX + dy * viewY - eyeX;
    float dirY = -dx * viewZ + dy * viewY - eyeY;
    float dirZ = dx * viewX - dy * viewZ - eyeZ;

    float len = sqrt(dirX * dirX + dirY * dirY + dirZ * dirZ);
    dirX /= len;
    dirY /= len;
    dirZ /= len;

    float zX = 0.0;
    float zY = 0.0;
    float zZ = 0.0;

    int iteration;
    for (iteration = 0; iteration < ITERATIONS && (zX * zX + zY * zY + zZ * zZ) < 2.0; iteration++) {
        float r = sqrt(zX * zX + zY * zY + zZ * zZ);
        float theta = acos(zZ / r);
        float phi = atan2(zY, zX);

        float zr = pow(r, 8.0);
        float ztheta = 8.0 * theta;
        float zphi = 8.0 * phi;

        zX = zr * sin(ztheta) * cos(zphi) + dirX;
        zY = zr * sin(ztheta) * sin(zphi) + dirY;
        zZ = zr * cos(ztheta) + dirZ;
    }

    int color = iteration * COLOR_SCALE + COLOR_OFFSET;
    image[y * width + x] = (color << 16) | (color << 8) | color;
}

int main() {
    int* image;
    cudaMalloc((void**)&image, WIDTH * HEIGHT * sizeof(int));

    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);
    MandelbulbKernel<<<gridSize, blockSize>>>(image, WIDTH, HEIGHT, EYE_X, EYE_Y, EYE_Z, VIEW_X, VIEW_Y, VIEW_Z);

    cudaDeviceSynchronize();

    // Copy image from device to host
    int* hostImage = new int[WIDTH * HEIGHT];
    cudaMemcpy(hostImage, image, WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);

    // Save the image to a bitmap file
    std::ofstream file("mandelbulb.bmp", std::ios::binary | std::ios::out);
    if (file.is_open()) {
        BMPHeader header;
        header.type = 0x4D42;
        header.size = WIDTH * HEIGHT * 3 + sizeof(BMPHeader) + sizeof(BMPInfoHeader);
        header.reserved1 = 0;
        header.reserved2 = 0;
        header.offset = sizeof(BMPHeader) + sizeof(BMPInfoHeader);
        file.write((char*)&header, sizeof(BMPHeader));

        BMPInfoHeader info;
        info.size = sizeof(BMPInfoHeader);
        info.width = WIDTH;
        info.height = HEIGHT;
        info.planes = 1;
        info.bits = 24;
        info.compression = 0;
        info.imagesize = WIDTH * HEIGHT * 3;
        info.xresolution = 0;
        info.yresolution = 0;
        info.ncolours = 0;
        info.importantcolours = 0;
        file.write((char*)&info, sizeof(BMPInfoHeader));

        for (int y = HEIGHT - 1; y >= 0; y--) {
            for (int x = 0; x < WIDTH; x++) {
                int color = hostImage[y * WIDTH + x];
                file.put((color >> 16) & 0xff);
                file.put((color >> 8) & 0xff);
                file.put(color & 0xff);
            }
        }

        file.close();
    } else {
       
        std::cerr << "Error opening file 'mandelbulb.bmp' for writing" << std::endl;
        return 1;
    }

    delete[] hostImage;
    cudaFree(image);
    return 0;
}
