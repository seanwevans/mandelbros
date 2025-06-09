#include <immintrin.h>  // AVX/AVX2 intrinsics
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "bmp.h"

#define WIDTH 1920
#define HEIGHT 1080
#define MAX_ITER 256
#define SCALE_X 3.5f
#define SCALE_Y 2.0f
#define OFFSET_X -2.5f
#define OFFSET_Y -1.0f

void mandelbrot_simd(uint32_t* image) {
    const __m256 scale_x = _mm256_set1_ps(SCALE_X / WIDTH);
    const __m256 scale_y = _mm256_set1_ps(SCALE_Y / HEIGHT);
    const __m256 offset_x = _mm256_set1_ps(OFFSET_X);
    const __m256 offset_y = _mm256_set1_ps(OFFSET_Y);
    const __m256 threshold = _mm256_set1_ps(4.0f);
    const __m256i one = _mm256_set1_epi32(1);

    for (int py = 0; py < HEIGHT; ++py) {
        __m256 y0 = _mm256_add_ps(_mm256_set1_ps(py), _mm256_set1_ps(0.5f));
        y0 = _mm256_mul_ps(y0, scale_y);
        y0 = _mm256_add_ps(y0, offset_y);

        for (int px = 0; px < WIDTH; px += 8) {
            __m256 x0 = _mm256_add_ps(_mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0), _mm256_set1_ps(px));
            x0 = _mm256_mul_ps(x0, scale_x);
            x0 = _mm256_add_ps(x0, offset_x);

            __m256 x = x0;
            __m256 y = y0;

            __m256i iter = _mm256_setzero_si256();
            for (int i = 0; i < MAX_ITER; ++i) {
                __m256 x2 = _mm256_mul_ps(x, x);
                __m256 y2 = _mm256_mul_ps(y, y);
                __m256 xy = _mm256_mul_ps(x, y);

                __m256 mag = _mm256_add_ps(x2, y2);
                __m256 mask = _mm256_cmp_ps(mag, threshold, _CMP_LT_OQ);

                int done = _mm256_movemask_ps(mask);
                if (done == 0) break;

                iter = _mm256_add_epi32(iter, _mm256_and_si256(one, _mm256_castps_si256(mask)));
                x = _mm256_add_ps(_mm256_sub_ps(x2, y2), x0);
                y = _mm256_add_ps(_mm256_add_ps(xy, xy), y0);
            }

            for (int i = 0; i < 8; ++i) {
                int index = py * WIDTH + px + i;
                int color = _mm256_extract_epi32(iter, i);
                int gray = 255 - (color * 255 / MAX_ITER);
                image[index] = gray | (gray << 8) | (gray << 16);
            }
        }
    }
}

int main() {
    uint32_t* image = malloc(WIDTH * HEIGHT * sizeof(uint32_t));
    if (!image) {
        fprintf(stderr, "Failed to allocate memory for image.\n");
        return 1;
    }

    mandelbrot_simd(image);
    write_bmp("mandelbrot.bmp", image, WIDTH, HEIGHT);

    free(image);
    return 0;
}
