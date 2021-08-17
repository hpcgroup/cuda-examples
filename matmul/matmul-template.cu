#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>

/** Compute C = A.B + C
 *  where A is MxP, B is PxN, and C is MxN
 */
__global__ void matmul(double *C, double *A, double *B, size_t M, size_t P, size_t N) {

    /* TODO -- define matmul */

}

int main() {

    double *h_C, *h_A, *h_B;
    double *d_C, *d_A, *d_B;
    const size_t M = 500, P = 500, N = 650;

    h_C = new double[M*N];
    h_A = new double[M*P];
    h_B = new double[P*N];

    for (size_t i = 0; i < M; i += 1) {
        for (size_t j = 0; j < P; j += 1) {
            h_A[i*P+j] = (i==j) ? 1.0 : 0.0;
        }
    }
    for (size_t i = 0; i < P*N; i += 1) {
        h_B[i] = 42.0;
    }

    cudaMalloc(&d_C, M*N*sizeof(double));
    cudaMalloc(&d_A, M*P*sizeof(double));
    cudaMalloc(&d_B, P*N*sizeof(double));

    cudaMemcpy(d_A, h_A, M*P*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, P*N*sizeof(double), cudaMemcpyHostToDevice);

    delete[] h_A;
    delete[] h_B;

    /* TODO -- add grid and block dimensions */
    dim3 threadsPerBlock (1,1);
    dim3 numBlocks (1,1);
    matmul<<<numBlocks, threadsPerBlock>>>(d_C, d_A, d_B, M, P, N);


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
