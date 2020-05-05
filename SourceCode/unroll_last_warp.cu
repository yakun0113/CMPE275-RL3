#include <iostream>
#include <math.h>

__device__ void warpReduce(volatile int* sdata, int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

__global__ void reduce2(int *g_idata, int *g_odata) {
  extern __shared__ int sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = g_idata[i];
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid < 32) warpReduce(sdata, tid);

  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int main(void)
{
  int N = 1<<28;

   int *input, *output;
   cudaMallocManaged(&input, N * sizeof(int));
   cudaMallocManaged(&output, N * sizeof(int));

  for (int i = 0; i < N; i++) {
    input[i] = 2;
    output[i] = 0;
  }

  int blockSize = 128;
  int numBlocks = (N + blockSize - 1) / blockSize;
  int smemSize = blockSize * sizeof(int);

  reduce2<<<numBlocks, blockSize, smemSize>>>(input, output);

  cudaDeviceSynchronize();

  int final_result = 0;
  for (int i = 0; i < numBlocks; i++) {
    final_result += output[i];
  }
  std::cout << "final result = " << final_result << "\n";

  // Free memory
  cudaFree(input);
  cudaFree(output);

  return 0;
}
