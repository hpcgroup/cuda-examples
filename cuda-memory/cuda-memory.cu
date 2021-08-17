#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>


int main() {

    double *h_z, *h_x, *h_y;
    double *d_z, *d_x, *d_y;
    const size_t N = 100;

    h_z = new double[N];
    h_x = new double[N];
    h_y = new double[N];

    for (size_t i = 0; i < N; i += 1) {
        h_x[i] = rand();
        h_y[i] = 1.0 / (h_x[i] + 1E8);
    }

    cudaMalloc(&d_z, N);
    cudaMalloc(&d_x, N);
    cudaMalloc(&d_y, N);

    cudaMemcpy(d_x, h_x, N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N, cudaMemcpyHostToDevice);
    delete[] h_x;
    delete[] h_y;

    
    /* ... */
    

    cudaMemcpy(h_z, d_z, N, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < N; i += 1) {
        std::cout << h_z[i] << " ";
    }
    std::cout << "\n";

    delete[] h_z;

    cudaFree(d_z);
    cudaFree(d_x);
    cudaFree(d_y);
}
