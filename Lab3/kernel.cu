
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define en 0.0002f
#define p 0.5f
#define G 0.75f
#define N 4
#define NUM_THREADS 16
#define NUM_BLOCKS 1

int tto(int i, int j) {
	return i* N + j;
}

void sideUpdate(int i, int j, float* ua) {
	int idx = i * N + j;
	if (i == 0) ua[idx] = G * ua[1*N+j];
	else if (j == 0) ua[idx] = G * ua[i*N+1];
	else if (i == (N - 1)) ua[idx] = G * ua[(N-2)*N +  j];
	else if (j == (N - 1)) ua[idx] = G * ua[i*N+(N-2)];

}

void cornerUpdate(int i, int j, float* ua) {
	int idx = i * N + j;
	if (i == 0 && j == 0) ua[idx] = G * ua[1*N+j];
	else if (i == (N - 1) && j == 0)  ua[idx] = G * ua[(N-2)*N+j];
	else if (i == 0 && j == (N - 1)) ua[idx] = G * ua[i*N+(N-2)];
	else if (i == (N - 1) && j == (N - 1)) ua[idx] = G * ua[(N-1)*N + (N-2)];

}

void interiorUpdate(int i, int j, float* ua, float* ub, float* uc) {
	int idx = i * N + j;
	ua[idx] = (p * (ub[(i-1)*N+j] + ub[(i + 1) * N + j] + ub[(i * N) + j + 1] + ub[(i * N) + j - 1] - (4 * ub[idx])) + 2*ub[idx] - (1 - en) * uc[idx]) / (1 + en);
}

__global__ void interiorUpdateGPU(float* ua, float* ub, float* uc) {
	int idx = (blockIdx.x * blockDim.x + threadIdx.x);
	for (int k = idx; k < N * N; k += NUM_THREADS) {
		int i = k / N; 
		int j = k % N; 
		if (i >= 1 && i <= N - 2 && j >= 1 && j <= N - 2) ua[idx] = (p * (ub[(i - 1) * N + j] + ub[(i + 1) * N + j] + ub[(i * N) + j + 1] + ub[(i * N) + j - 1] - (4 * ub[idx])) + 2 * ub[idx] - (1 - en) * uc[idx]) / (1 + en);
	}
}

__global__ void sideUpdateGPU(float* u) {
	int idx = (blockIdx.x * blockDim.x + threadIdx.x);
	for (int k = idx; k < N * N; k += NUM_THREADS) {
		int i = k / N; 
		int j = k % N;
		if (i == 0) u[idx] = G * u[1 * N + j];
		else if (j == 0) u[idx] = G * u[i * N + 1];
		else if (i == (N - 1)) u[idx] = G * u[(N - 2) * N + j];
		else if (j == (N - 1)) u[idx] = G * u[i * N + (N - 2)];
	}
}

__global__ void cornerUpdateGPU(float* u) {
	int idxB = (blockIdx.x * blockDim.x + threadIdx.x);
	for (int k = idxB; k < N * N; k += NUM_THREADS) {
		int i = k / N;
		int j = k % N;
		int idx = i * N + j;
		if (i == 0 && j == 0) u[idx] = G * u[1 * N + j];
		else if (i == (N - 1) && j == 0)  u[idx] = G * u[(N - 2) * N + j];
		else if (i == 0 && j == (N - 1)) u[idx] = G * u[i * N + (N - 2)];
		else if (i == (N - 1) && j == (N - 1)) u[idx] = G * u[(N - 1) * N + (N - 2)];
	}
}

void parallelDrum(int iterations, int fds, float* u, float* u1, float* u2) {

	int k = 0;
	while (k < iterations){

		interiorUpdateGPU << < NUM_BLOCKS, NUM_THREADS >> > (u, u1, u2);
		cudaDeviceSynchronize();
		sideUpdateGPU << < NUM_BLOCKS, NUM_THREADS >> > (u);
		cudaDeviceSynchronize();
		cornerUpdateGPU << < NUM_BLOCKS, NUM_THREADS >> > (u);
		cudaDeviceSynchronize();

		memcpy(u2, u1, fds);
		memcpy(u1, u, fds);

		printf("u[%d][%d] = %.6f\n", N / 2, N / 2, u[tto(N/2,N/2)]);
		k += 1;
	}

}

void serialDrum(int iterations, int dataSize, float* u, float* u1, float* u2) {
	int k = 0;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			printf("%.6f,", u[tto(i, j)]);
		}
		printf("\n");
	}
	printf("---------------------------------\n\n");
	while (k < iterations) {
		for (int i = 1; i < N - 1; i++) {
			for (int j = 1; j < N - 1; j++) {
				interiorUpdate(i, j, u, u1, u2);
			}
		}

		for (int i = 1; i < N - 1; i++) {
			sideUpdate(0, i, u);
			sideUpdate(N - 1, i, u);
			sideUpdate(i, 0, u);
			sideUpdate(i, N - 1, u);
		}

		cornerUpdate(0, 0, u);
		cornerUpdate(N - 1, 0, u);
		cornerUpdate(0, N - 1, u);
		cornerUpdate(N - 1, N - 1, u);

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				printf("%.6f,", u[tto(i, j)]);
			}
			printf("\n");
		}
		printf("---------------------------------\n\n");
		memcpy(u2, u1, dataSize * sizeof(float));
		memcpy(u1, u, dataSize * sizeof(float));
		k += 1;
	}
}


int main()
{
    int numIts = 4;//atoi(argv[1]);
	int dataSize = N * N;
	int fds = sizeof(float) * dataSize;

	float* u = (float*)calloc(dataSize,sizeof(float));
	float* u1 = (float*)calloc(dataSize,sizeof(float));
	float* u2 = (float*)calloc(dataSize,sizeof(float));

	if (u == NULL) exit(0);
	if (u1 == NULL) exit(0);
	if (u2 == NULL) exit(0);

	u1[tto(N / 2, N / 2)] = 1.f;

	serialDrum(numIts, dataSize,u,u1,u2);

	cudaMallocManaged((void**)& u, fds);
	cudaMallocManaged((void**)& u1, fds);
	cudaMallocManaged((void**)& u2, fds);

	u1[tto(N / 2, N / 2)] = 1.f;

	parallelDrum(numIts, fds, u, u1, u2);


	cudaFree(u);
	cudaFree(u1);
	cudaFree(u2);

}

