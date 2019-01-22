#pragma once

#include <iostream>
#include <stdio.h>
#include <chrono>
#include <cstring>
#include <ctime>

__host__ __device__
bool print_page(int *btree);

__host__ __device__
bool print_btree(int *btree, int N);

__device__ 
uint32_t get_warp_id();

__device__ 
uint32_t get_local_warp_id();

__host__ __device__
bool print_page(int *btree)
{
    printf("Obecna strona: ");
    for(int k = 0; k < 3; k++)
    {
        printf("%d ", btree[k]);
    }
    printf("\n");
}

__host__ __device__
bool print_btree(int *btree, int N)
{
    printf("\nB+Drzewo: ");
    for(int k = 0; k < N; k++)
    {
        if(btree[k] != 0)
        {
            printf("%d ", btree[k]);
        }
    }

    printf("\n");
}

__device__ 
uint32_t get_warp_id()
{
	int tid = threadIdx.x + 
        threadIdx.y * blockDim.x + 
        threadIdx.z * blockDim.x * blockDim.y +
        blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
    
    return tid / 32;
}

__device__ 
uint32_t get_local_warp_id()
{
    int tid = threadIdx.x + 
        threadIdx.y * blockDim.x + 
        threadIdx.z * blockDim.x * blockDim.y;

    return tid / 32;
}