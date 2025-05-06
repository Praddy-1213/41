#include <stdio.h>
#include <stdlib.h>

#define N 4  // Define the size of the matrix

__global__ void matrixMultiply(int* A, int* B, int* C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        int sum = 0;
        for (int k = 0; k < size; k++) {
            sum += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = sum;
    }
}

void initialize(int* matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = rand() % 10;  // Initialize with random values
between 0 and 9
    }
}

void print(int* matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%d ", matrix[i * size + j]);
        }
        printf("\n");
    }
}

int main() {
    int *A, *B, *C;  // Pointers for matrices A, B, and C

    size_t matrixBytes = N * N * sizeof(int);  // Size of the matrices in bytes

    // Allocate memory on the host for matrices A, B, and C
    A = (int*)malloc(matrixBytes);
    B = (int*)malloc(matrixBytes);
    C = (int*)malloc(matrixBytes);

    // Initialize matrices A and B with random values
    initialize(A, N);
    initialize(B, N);

    printf("Matrix A:\n");
    print(A, N);
    printf("\nMatrix B:\n");
    print(B, N);

    int *d_A, *d_B, *d_C;

    // Allocate memory on the device (GPU) for matrices A, B, and C
    cudaMalloc(&d_A, matrixBytes);
    cudaMalloc(&d_B, matrixBytes);
    cudaMalloc(&d_C, matrixBytes);

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, A, matrixBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, matrixBytes, cudaMemcpyHostToDevice);

    // Set the block and grid sizes
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) /
threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy the result matrix C from device to host
    cudaMemcpy(C, d_C, matrixBytes, cudaMemcpyDeviceToHost);

    printf("\nMatrix C (Result of A * B):\n");
    print(C, N);

    // Free allocated memory on the host
    free(A);
    free(B);
    free(C);

    // Free allocated memory on the device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}