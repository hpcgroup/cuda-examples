#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>

/** Compute C = A + B
 *  where A,B,C are MxN
 */
__global__ void matrixAdd(double *C, double *A, double *B, size_t M, size_t N) {

    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int j = blockDim.y*blockIdx.y + threadIdx.y;
    if (i < M && j < N) {
        int idx = i*N+j;
        C[idx] = A[idx] + B[idx];
    }
}

int main() {

    double *h_C, *h_A, *h_B;
    double *d_C, *d_A, *d_B;
    const size_t M = 500, N = 650;

    h_C = new double[M*N];
    h_A = new double[M*N];
    h_B = new double[M*N];

    for (size_t i = 0; i < M*N; i += 1) {
        h_A[i] = 19.5;
        h_B[i] = 22.5;
    }

    cudaMalloc(&d_C, M*N*sizeof(double));
    cudaMalloc(&d_A, M*N*sizeof(double));
    cudaMalloc(&d_B, M*N*sizeof(double));

    cudaMemcpy(d_A, h_A, M*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, M*N*sizeof(double), cudaMemcpyHostToDevice);

    delete[] h_A;
    delete[] h_B;

    dim3 threadsPerBlock (16, 16);
    dim3 numBlocks (
        M/threadsPerBlock.x + (M%threadsPerBlock.x != 0),
        N/threadsPerBlock.y + (N%threadsPerBlock.y != 0));
    matrixAdd<<<numBlocks, threadsPerBlock>>>(d_C, d_A, d_B, M, N);


    cudaMemcpy(h_C, d_C, M*N*sizeof(double), cudaMemcpyDeviceToHost);

    bool success = true;
    for (size_t i = 0; i < M*N; i += 1) {
        if (std::abs(h_C[i] - (42.0)) > 1E-8)
            success = false;
    }
    if (!success)
        fprintf(stderr, "Error: incorrect results!\n");

    delete[] h_C;

    cudaFree(d_C);
    cudaFree(d_A);
    cudaFree(d_B);
}
