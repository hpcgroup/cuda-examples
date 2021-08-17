#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>

__global__ void saxpy(double *z, double *x, double *y, double alpha, size_t N) {

    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < N)
        z[i] = alpha*x[i] + y[i];
}

int main() {

    double *h_z, *h_x, *h_y;
    double *d_z, *d_x, *d_y;
    double alpha = 1.5;
    const size_t N = 10000;

    h_z = new double[N];
    h_x = new double[N];
    h_y = new double[N];

    for (size_t i = 0; i < N; i += 1) {
        h_x[i] = 5.0;
        h_y[i] = -2.0;
    }

    cudaMalloc(&d_z, N*sizeof(double));
    cudaMalloc(&d_x, N*sizeof(double));
    cudaMalloc(&d_y, N*sizeof(double));

    cudaMemcpy(d_x, h_x, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N*sizeof(double), cudaMemcpyHostToDevice);

    delete[] h_x;
    delete[] h_y;

    size_t threadsPerBlock = 512;
    size_t numBlocks = N/threadsPerBlock + (N % threadsPerBlock != 0);
    saxpy<<<numBlocks, threadsPerBlock>>>(d_z, d_x, d_y, alpha, N);


    cudaMemcpy(h_z, d_z, N*sizeof(double), cudaMemcpyDeviceToHost);

    bool success = true;
    for (size_t i = 0; i < N; i += 1) {
        if (std::abs(h_z[i] - (1.5*5.0-2.0)) > 1E-8)
            success = false;
    }
    if (!success)
        fprintf(stderr, "Error: incorrect results!\n");

    delete[] h_z;

    cudaFree(d_z);
    cudaFree(d_x);
    cudaFree(d_y);
}
