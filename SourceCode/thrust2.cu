#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>

#include <iostream>

int main(void)
{

    const int ARRAY_SIZE = 1<<28;

    int* mA;
    cudaMallocManaged(&mA, ARRAY_SIZE * sizeof(int));

    for (int i = 0; i < ARRAY_SIZE; i++) {
      mA[i] = 2;
    }

    int sum =
      thrust::reduce(
        thrust::cuda::par,
        mA, mA + ARRAY_SIZE,
        (int) 0,
        thrust::plus<int>());

    cudaDeviceSynchronize();

    std::cout << "sum = " << sum << std::endl;

    cudaFree(mA);
    return 0;
}
