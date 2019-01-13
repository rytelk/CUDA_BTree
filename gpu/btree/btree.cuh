#pragma once

#include <iostream>
#include <stdio.h>
#include <chrono>
#include <cstring>
#include <ctime>

__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

__host__ 
void gpu_btree_test()
{
    std::cout << "\nB+Tree GPU" << std::endl;    
    int N = 1<<20;
    float *x, *y, *d_x, *d_y;
    
    float *x = new float[N];
    float *y = new float[N];

    cudaMalloc(&d_x, N*sizeof(float)); 
    cudaMalloc(&d_y, N*sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

    add<<<(N+255)/256, 256>>>(N, d_x, d_y);

    cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = max(maxError, abs(y[i]-4.0f));
    printf("Max error: %f\n", maxError);

    cudaFree(d_x);
    cudaFree(d_y);
    delete [] x;
    delete [] y;
}