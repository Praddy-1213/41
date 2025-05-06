#include <stdio.h>
#include <stdlib.h>
#define N 4 // Matrix dimension size

// CUDA kernel for matrix-vector multiplication
__global__

void matrixVecMul(int* A, int* B, int* C, int size) 
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < size) 
    {
        int sum = 0;
        for (int k = 0; k < size; k++) 
        {
            sum += A[row * size + k] * B[k];
        }
        C[row] = sum;
    }
}

// Function to initialize a matrix or vector with random values
void initialize(int* vector, int size) 
{
    for (int i = 0; i < size; i++) 
    {
        vector[i] = rand() % 10; // Random values between 0 and 9
    }
}

// Function to print a vector
void print(int* vector, int size) 
{
    for (int i = 0; i < size; i++) 
    {
        printf("%d ", vector[i]);
    }
    printf("\n");
}

// Function to print a matrix
void printMatrix(int* matrix, int size) 
{
    for (int i = 0; i < size; i++) 
    {
        for (int j = 0; j < size; j++) 
        {
            printf("%d ", matrix[i * size + j]);
        }
    printf("\n");
    }
}

int main() 
{
    int* A, * B, * C;
    size_t matrixBytes = N * N * sizeof(int);
    size_t vectorBytes = N * sizeof(int);
    
    // Allocate memory for matrix and vectors
    A = (int*)malloc(matrixBytes);
    B = (int*)malloc(vectorBytes);
    C = (int*)malloc(vectorBytes);
    int* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, matrixBytes);
    cudaMalloc(&d_B, vectorBytes);
    cudaMalloc(&d_C, vectorBytes);

    // Initialize matrix A and vector B
    initialize(A, N * N); // A is an NxN matrix
    initialize(B, N); // B is a vector of size N

    // Print matrix A and vector B
    printf("Matrix A:\n");
    printMatrix(A, N);
    printf("Vector B:\n");
    print(B, N);

    // Copy data from host to device
    cudaMemcpy(d_A, A, matrixBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, vectorBytes, cudaMemcpyHostToDevice);

    // Define the number of threads and blocks
    int threadsPerBlock = 16;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the matrix-vector multiplication kernel
    matrixVecMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C,N);
    
    // Copy the result back to the host
    cudaMemcpy(C, d_C, vectorBytes, cudaMemcpyDeviceToHost);

    // Print the result of the multiplication
    printf("Matrix-Vector multiplication result (C = A * B):\n");
    print(C, N);

    // Free allocated memory
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

//commands to run:
// vim multiply.cu
// nvcc multiply.cu -o multiply
// ./multiply