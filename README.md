# Mandelbrot Examples

This repository contains several small programs for generating Mandelbrot set
images using different languages and technologies. The code was taken from a
variety of experiments and is not guaranteed to be production ready, but it can
serve as a starting point for exploring CPU, GPU and JavaScript implementations.

## Requirements

To build the examples you will need:

- **gcc** – for the plain C implementation using AVX intrinsics.
- **nvcc** (CUDA Toolkit) – for the CUDA examples.
- **gmp** development headers and library – required by the CUDA/C++ versions
  that use the GNU Multiple Precision Arithmetic Library (`<gmp.h>`).
- A C++ compiler such as **g++** (generally provided with the CUDA Toolkit when
  using `nvcc`).

The C program also includes a header `bmp.h` used for writing out an image.
You will need to provide this header (and any corresponding implementation) or
replace the image output logic if you want to build `mandelbrot.c`.

## Building

### mandelbrot.c

This is a SIMD AVX implementation. Compile it with gcc and enable AVX/AVX2
instructions:

```bash
gcc -O2 -mavx2 -std=c11 mandelbrot.c -o mandelbrot_c -lm
```

`bmp.h` must be available at compile time. Running the program will create a
`mandelbrot.bmp` file.

### mandelbrot.cu

CUDA version that also relies on GMP for arbitrary precision arithmetic.
Compile it with `nvcc` and link against the GMP library:

```bash
nvcc -O2 mandelbrot.cu -o mandelbrot_cuda -lgmp
```

### mandelbrot.cpp

Although this file has a `.cpp` extension it makes use of CUDA kernels. It can
also be compiled with `nvcc`:

```bash
nvcc -O2 mandelbrot.cpp -o mandelbrot_cpp_cuda -lgmp
```

### mandelbulb.cu

Another CUDA example that renders a Mandelbulb fractal:

```bash
nvcc -O2 mandelbulb.cu -o mandelbulb_cuda
```

## Viewing `mandelbrot.html`

`mandelbrot.html` demonstrates drawing the Mandelbrot set using JavaScript.
Open the file directly in any modern web browser or serve the repository using
a simple web server and navigate to the page:

```bash
python3 -m http.server
```

Then open <http://localhost:8000/mandelbrot.html> in your browser.


## License

This project is licensed under the [MIT License](LICENSE).

