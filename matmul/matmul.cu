#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 32;

/** Compute C = A.B + C
 *  where A is MxP, B is PxN, and C is MxN
 */
__global__ void matmul(double *C, double *A, double *B, size_t M, size_t P, size_t N) {

    float sum = 0.0;
    int x = threadIdx.x;
    int y = threadIdx.y;

    /* get pointer to blockIdx.y, blockIdx.x sub-array of C */
    double *cBlock = &C[blockIdx.y*BLOCK_SIZE*N + BLOCK_SIZE*blockIdx.x]; 

    for (int i = 0; i < P/BLOCK_SIZE; i += 1) {

        /* get pointer to blockIdx.y, i sub-array of A */
        double *aBlock = &A[blockIdx.x*BLOCK_SIZE*P + BLOCK_SIZE*i];

        /* get pointer to i, blockIdx.x sub-array of B */
        double *bBlock = &B[i*BLOCK_SIZE*N + BLOCK_SIZE*blockIdx.x];

        __shared__ double aShared[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double bShared[BLOCK_SIZE][BLOCK_SIZE];

        aShared[y][x] = aBlock[y*P+x];
        bShared[y][x] = bBlock[y*N+x];

        __syncthreads();

        /* multiply the sub-arrays together */
        for (int j = 0; j < BLOCK_SIZE; j += 1) {
            sum += aShared[y][j] * bShared[j][x];
        }

        __syncthreads();
    }

    cBlock[y*N+x] = sum;
}

int main() {

    double *h_C, *h_A, *h_B;
    double *d_C, *d_A, *d_B;
    const size_t M = 512, P = 512, N = 256;

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

    dim3 threadsPerBlock (BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks (
        N/threadsPerBlock.x + (M%threadsPerBlock.x != 0),
        M/threadsPerBlock.y + (N%threadsPerBlock.y != 0));
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
