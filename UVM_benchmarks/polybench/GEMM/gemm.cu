/**
 * gemm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>

#include "../../../common/polybenchUtilFuncts.h"

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)


#define GPU_DEVICE 0

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 32412.0f
#define BETA 2123.0f

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void gemm(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, int NI, int NJ, int NK)
{
	int i,j,k;
	
	for (i = 0; i < NI; i++)
	{
    	for (j = 0; j < NJ; j++)
    	{
			C[i*NJ + j] *= BETA;
	
			for (k = 0; k < NK; ++k)
			{
	  			C[i*NJ + j] += ALPHA * A[i*NK + k] * B[k*NJ + j];
			}
      	}
	}
}


void init(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, int NI, int NJ, int NK)
{
	int i, j;

  	for (i = 0; i < NI; i++)
	{
    	for (j = 0; j < NK; j++)
		{
			A[i*NK + j] = ((DATA_TYPE) i*j) / NI;
		}
	}

  	for (i = 0; i < NK; i++)
	{
    	for (j = 0; j < NJ; j++)
		{
			  B[i*NJ + j] = ((DATA_TYPE) i*j + 1) / NJ;
		}
	}

  	for (i = 0; i < NI; i++)
	{
    	for (j = 0; j < NJ; j++)
		{
			  C[i*NJ + j] = ((DATA_TYPE) i*j + 2) / NJ;
		}
	}
}


void compareResults(DATA_TYPE* C, DATA_TYPE* C_outputFromGpu, int NI, int NJ, int NK)
{
	int i, j, fail;
	fail = 0;
	
	// Compare C1 and C2
	for (i=0; i < NI; i++) 
	{
		for (j=0; j < NJ; j++) 
		{
			if (percentDiff(C[i*NJ + j], C_outputFromGpu[i*NJ + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init(int* device_count)
{
	cudaGetDeviceCount(device_count);
	if (device_count == 0) {
		fprintf(stderr, "There are no device available\n");
		exit(EXIT_FAILURE);
	}

	for (int device=0; device < (*device_count); device++) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		printf("Setting device %d with name %s\n", device, deviceProp.name);
	}
}

void GPU_p2p_enable(int device_count) 
{
	for (int device=0; device < device_count; device++) {
		for (int peer=0; peer < device_count; peer++) {
			if (device == peer) 
				continue;
			int peerEnable = 0;
			cudaDeviceCanAccessPeer(&peerEnable, device, peer);

			if (!peerEnable) {
				printf("Device %d cannot access peer device\n", device);
				exit(EXIT_FAILURE);
			}
		}
	}

	for (int device=0; device < device_count; device++) {
		cudaSetDevice(device);
		for (int peer=0; peer < device_count; peer++) {
			if (device == peer) 
				continue;
			cudaDeviceEnablePeerAccess(peer, 0);
			printf("Device %d -> Peer Device %d\n", device, peer);
		}
	}
}

__global__ void gemm_kernel(DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *c, int NI, int NJ, int NK)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NI) && (j < NJ))
	{	
		c[i * NJ + j] *= BETA;
		int k;
		for(k=0; k < NK; k++)
		{
			c[i * NJ + j] += ALPHA * a[i * NK + k] * b[k * NJ +j];
		}
	}
}


void gemmCuda(DATA_TYPE* A_gpu, DATA_TYPE* B_gpu, DATA_TYPE* C_gpu, int NI, int NJ, int NK, int device_count)
{
	double t_start, t_end;

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)(ceil( ((float)NI)/ ((float)block.x) )),(size_t)(ceil( ((float)NJ)/ ((float)block.y) )));

	int rows_per_device = NI / device_count;
	int remaining_rows = NI % device_count;

	t_start = rtclock();

	for (int device=0; device < device_count; device++) {
		cudaSetDevice(device);

		int start_row = device * rows_per_device;
		int current_rows = rows_per_device;

		if (device == device_count - 1) 
			current_rows += remaining_rows;

		DATA_TYPE *A_gpu_sub = A_gpu + start_row * NK;
		DATA_TYPE *B_gpu_sub = B_gpu;
		DATA_TYPE *C_gpu_sub = C_gpu + start_row * NJ;

		gemm_kernel<<< grid, block >>>(A_gpu_sub, B_gpu_sub, C_gpu_sub, NI, NJ, NK);
	}

	for (int device=0; device < device_count; device++) {
		cudaSetDevice(device);
		cudaDeviceSynchronize();
	}

	t_end = rtclock();

	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);   
}
	
int main(int argc, char *argv[])
{
	int NI, NJ, NK = 64;
	if (argc >= 2) NI = atoi(argv[1]);
	if (argc >= 3) NJ = atoi(argv[2]);
	if (argc >= 4) NK = atoi(argv[3]);
	
#ifdef TEST
	double t_start, t_end;

	DATA_TYPE* C;
	C = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE)); 
#endif // TEST

	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu; 

	cudaMallocManaged(&A_gpu, sizeof(DATA_TYPE) * NI * NK);
	cudaMallocManaged(&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
	cudaMallocManaged(&C_gpu, sizeof(DATA_TYPE) * NI * NJ);
	
	init(A_gpu, B_gpu, C_gpu, NI, NJ, NK);
	
	int device_count = 0;
	GPU_argv_init(&device_count);
	GPU_p2p_enable(device_count);
	
	gemmCuda(A_gpu, B_gpu, C_gpu, NI, NJ, NK, device_count);
	printf("GEMM Done\n");

#ifdef TEST
	t_start = rtclock();
	gemm(A_gpu, B_gpu, C, NI, NJ, NK);
	t_end = rtclock();

	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	
	compareResults(C, C_gpu, NI, NJ, NK);

	free(C);
#endif // TEST

	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);
    return 0;
}

