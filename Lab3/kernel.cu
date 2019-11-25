
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <string.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

#define en 0.0002
#define p 0.5
#define G 0.75


void sideUpdate(int i, int j, int N, float** u, float** u1, float** u2) {
	if (i == 0) u[i][j] = G * u[1][j];
	else if (j == 0) u[i][j] = G * u[i][1];
	else if (i == (N - 1)) u[i][j] = G * u[N - 2][j];
	else if (j == (N - 1)) u[i][j] = G * u[i][N - 2];

}

void cornerUpdate(int i, int j, int N, float** u, float** u1, float** u2) {
	if (i == 0 && j == 0) u[i][j] = G * u[1][j];
	else if (i == (N - 1) && j == 0) u[i][j] = G * u[N - 2][j];
	else if (i == 0 && j == (N - 1)) u[i][j] = G * u[i][N - 2];
	else if (i == (N - 1) && j == (N - 1)) u[i][j] = G * u[N - 1][N - 2];

}

void interiorUpdate(int i, int j, float** u, float** u1, float** u2) {

	u[i][j] = (p * (u1[i - 1][j] + u1[i + 1][j] + u1[i][j + 1] + u1[i][j - 1] - (4 * u1[i][j])) + 2*u1[i][j] - (1 - en) * u2[i][j]) / (1 + en);

}

void initvalues(float* u, float* u1, float* u2) {


}

int main()
{

	float *u[4];
	float ua[4] = { 0.0, 0.0, .374925, .281194 },
		ub[4] = { 0.0, 0.0, .499900, .374925 },
		uc[4] = { .37492, .499900, 0.0, 0.0 },
		ud[4] = { .281194,  .374925, 0.0, 0.0 };


	int N = 4;
	u[0] = ua;
	u[1] = ub;
	u[2] = uc;
	u[3] = ud;


	float* u1[4];
	float* u2[4];
	memcpy(u1, u, 16 * sizeof(float));
	memcpy(u2, u, 16 * sizeof(float));

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			printf("%.6f,", u[i][j]);
		}
		printf("\n");
	}
	printf("---------------------------------\n\n");

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			printf("%.6f,", u1[i][j]);
		}
		printf("\n");
	}
	printf("---------------------------------\n\n");
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			printf("%.6f,", u2[i][j]);
		}
		printf("\n");
	}
	printf("---------------------------------\n\n");

	for (int i = 1; i < N-1; i++) {
		for (int j = 1; j < N-1; j++) {
			interiorUpdate(i, j, u, u1, u2);
		}
	}
	
	for (int i = 1; i < N - 1; i++) {
		sideUpdate(0, i, N, u, u1, u2);
		sideUpdate(N-1, i, N, u, u1, u2);
		sideUpdate(i, 0, N, u, u1, u2);
		sideUpdate(i, N-1, N, u, u1, u2);
	}

	cornerUpdate(0, 0, N, u, u1, u2);
	cornerUpdate(N-1, 0, N, u, u1, u2);
	cornerUpdate(0, N-1, N, u, u1, u2);
	cornerUpdate(N-1, N-1, N, u, u1, u2);
	
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			printf("%.6f,", u[i][j]);
		}
		printf("\n");
	}
	printf("---------------------------------\n\n");

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			printf("%.6f,", u1[i][j]);
		}
		printf("\n");
	}
	printf("---------------------------------\n\n");
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			printf("%.6f,", u2[i][j]);
		}
		printf("\n");
	}
	printf("---------------------------------\n\n");




		/*
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
	*/
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
