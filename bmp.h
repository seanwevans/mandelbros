#ifndef BMP_H
#define BMP_H

#include <stdint.h>
#include <stdio.h>

static inline int write_bmp(const char *path, uint32_t *data, int w, int h)
{
    FILE *f = fopen(path, "wb");
    if (!f) return 0;

    int row_size = (w * 3 + 3) & ~3;
    int img_size = row_size * h;
    unsigned char header[54] = {
        'B','M',     // signature
        0,0,0,0,     // file size
        0,0,0,0,     // reserved
        54,0,0,0,    // offset to pixel data
        40,0,0,0,    // header size
        0,0,0,0,     // width
        0,0,0,0,     // height
        1,0,         // planes
        24,0,        // bits per pixel
        0,0,0,0,     // compression (none)
        0,0,0,0,     // image size
        0,0,0,0,     // x pixels per meter
        0,0,0,0,     // y pixels per meter
        0,0,0,0,     // colors used
        0,0,0,0      // important colors
    };

    unsigned int file_size = 54 + img_size;
    header[2] = (unsigned char)(file_size);
    header[3] = (unsigned char)(file_size >> 8);
    header[4] = (unsigned char)(file_size >> 16);
    header[5] = (unsigned char)(file_size >> 24);

    header[18] = (unsigned char)(w);
    header[19] = (unsigned char)(w >> 8);
    header[20] = (unsigned char)(w >> 16);
    header[21] = (unsigned char)(w >> 24);

    header[22] = (unsigned char)(h);
    header[23] = (unsigned char)(h >> 8);
    header[24] = (unsigned char)(h >> 16);
    header[25] = (unsigned char)(h >> 24);

    header[34] = (unsigned char)(img_size);
    header[35] = (unsigned char)(img_size >> 8);
    header[36] = (unsigned char)(img_size >> 16);
    header[37] = (unsigned char)(img_size >> 24);

    fwrite(header, 1, 54, f);

    unsigned char pad[3] = {0, 0, 0};
    for (int y = h - 1; y >= 0; --y) {
        for (int x = 0; x < w; ++x) {
            uint32_t pixel = data[y * w + x];
            unsigned char rgb[3] = {
                (unsigned char)(pixel & 0xFF),
                (unsigned char)((pixel >> 8) & 0xFF),
                (unsigned char)((pixel >> 16) & 0xFF)
            };
            fwrite(rgb, 1, 3, f);
        }
        fwrite(pad, 1, row_size - w * 3, f);
    }

    fclose(f);
    return 1;
}

#endif // BMP_H
