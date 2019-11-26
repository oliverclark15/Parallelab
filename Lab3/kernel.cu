
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define en 0.0002
#define p 0.5
#define G 0.75
#define N 4

float u[4][4] = { {0} };
float u1[4][4];
float u2[4][4];


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int tto(int i, int j) {
	return i * N + j;
}


void sideUpdate(int i, int j, float* ua) {
	int idx = tto(i, j);
	if (i == 0) ua[idx] = G * ua[tto(1,j)];
	else if (j == 0) ua[idx] = G * ua[tto(i, 1)];
	else if (i == (N - 1)) ua[idx] = G * ua[tto(N-2, j)];
	else if (j == (N - 1)) ua[idx] = G * ua[tto(i, N-2)];

}

void cornerUpdate(int i, int j, float* ua) {
	int idx = tto(i, j);
	if (i == 0 && j == 0) ua[idx] = G * ua[tto(1,j)];
	else if (i == (N - 1) && j == 0)  ua[idx] = G * ua[tto(N-2,j)];
	else if (i == 0 && j == (N - 1)) ua[idx] = G * ua[tto(i,N-2)];
	else if (i == (N - 1) && j == (N - 1)) ua[idx] = G * ua[tto(N-1,N-2)];

}

void interiorUpdate(int i, int j, float* ua, float* ub, float* uc) {
	int idx = tto(i, j);
	ua[tto(i,j)] = (p * (ub[tto(i-1, j)] + ub[tto(i+1, j)] + ub[tto(i, j+1)] + ub[tto(i, j-1)] - (4 * ub[idx])) + 2*ub[idx] - (1 - en) * uc[idx]) / (1 + en);
}




int main()
{
    int numIts = 4;//atoi(argv[1]);
	int dataSize = N * N;

    float* u = (float*)malloc(dataSize * sizeof(float));
    float* u1 = (float*)malloc(dataSize * sizeof(float));
    float* u2 = (float*)malloc(dataSize * sizeof(float));
	int k = 0;
	for (int i = 0; i < dataSize; i++) {
		u[i] = 0.0;
		u1[i] = 0.0;
		u2[i] = 0.0;
	}
	u1[tto(N / 2, N / 2)] = 1;

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			printf("%.6f,", u[tto(i,j)]);
		}
		printf("\n");
	}
	printf("---------------------------------\n\n");
	while (k < numIts) {

		for (int i = 1; i < N - 1; i++) {
			for (int j = 1; j < N - 1; j++) {
				interiorUpdate(i, j, u, u1, u2);
			}
		}

		for (int i = 1; i < N - 1; i++) {
			sideUpdate(0, i, u);
			sideUpdate(N - 1, i,  u);
			sideUpdate(i, 0, u);
			sideUpdate(i, N - 1, u);
		}

		cornerUpdate(0, 0,u);
		cornerUpdate(N - 1, 0, u);
		cornerUpdate(0, N - 1, u);
		cornerUpdate(N - 1, N - 1, u);

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				printf("%.6f,", u[tto(i,j)]);
			}
			printf("\n");
		}
		printf("---------------------------------\n\n");
		memcpy(u2, u1, dataSize * sizeof(float));
		memcpy(u1, u, dataSize * sizeof(float));
		k += 1;
		
	}
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
