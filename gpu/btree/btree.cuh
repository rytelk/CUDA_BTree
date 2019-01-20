#pragma once

#include <iostream>
#include <stdio.h>
#include <chrono>
#include <cstring>
#include <ctime>

#include "mock_btree.cuh"
#include "btree_helper.cuh"

__global__
void search(int* keys, int keys_count, int *btree, int N, int degree, int levels_count);

__device__
bool check_leaf_key(int key, int index, int *btree);

__device__
bool check_page(int key, int index, int *btree, int degree);

__host__
void gpu_btree_test();

__global__
void search(int* keys, int keys_count, int *btree, int N, int degree, int levels_count)
{
    int index = threadIdx.x % 32;
    const int warps_per_block = 8;
    int local_warp_id = get_local_warp_id();
    int warp_id = get_warp_id();

    __shared__ int *current_level[8];
    __shared__ int *current_page[8];
    
    int stride = (blockDim.x * gridDim.x) / 32;
    int key;
    for(int i = warp_id; i < keys_count; i = i + stride)
    {
	if(index == 0)
  	    printf("GlobalWarpId: %d LocalWarpId: %d\n", warp_id, local_warp_id);
        key = keys[i];
        current_level[local_warp_id] = btree;
        current_page[local_warp_id] = btree;
        
        for(int level = levels_count; level >= 0; level--)
        {
            if(index < degree) 
            {
                if(level == 0)
                {
                    if(check_leaf_key(key, index, current_page[local_warp_id])) 
                    {
                        printf("Found key: %d\n", current_page[local_warp_id][index]);
                    }
                }
                else 
                {
                    if(check_page(key, index, current_page[local_warp_id], degree))
                    {
                        int *cur_elem = current_page[local_warp_id] + index;
                        int previous_elem_count = cur_elem - current_level[local_warp_id];
                        int offset = __powf(degree, levels_count - level) * degree;
                        current_level[local_warp_id] = current_level[local_warp_id] + offset;
                        current_page[local_warp_id] = current_level[local_warp_id] + previous_elem_count * degree;
                    }
                }
            }
            __syncthreads();
        }
    }
}

__device__
bool check_leaf_key(int key, int index, int *btree)
{
    return btree[index] == key;
}

__device__
bool check_page(int key, int index, int *btree, int degree)
{
    int local_warp_id = get_local_warp_id();
    int warp_id = get_warp_id();
    //printf("Warp Id: %d Local Warp Id: %d ThreadId: %d Key: %d Index: %d\n", warp_id, local_warp_id, index, key, index);
    if(index == 0) 
    {
        return btree[index] != -1 && key < btree[index];
    }
    if(btree[index-1] != -1 && btree[index] != -1)
    {
        //printf("2: Key: %d btree[i - 1]: %d btree[i]: %d\n", key, btree[index - 1], btree[index]);
        return key >= btree[index - 1] && key < btree[index];
    }
    if(btree[index] == -1)
    {
        //printf("3: Key: %d btree[i - 1]: %d\n", key, btree[index - 1]);
        return btree[index - 1] != -1 && key >= btree[index - 1];
    }
    
    return false; 
}

__host__
void gpu_btree_test()
{
    std::cout << "\nB+Tree GPU" << std::endl;

    int *btree;
    int N = 10000;
    int degree;
    int levels_count;
    
    btree = get_mock_btree2(N, degree, levels_count);
    print_btree(btree, N);

    int *d_btree;
    cudaMalloc(&d_btree, N*sizeof(int));
    cudaMemcpy(d_btree, btree, N*sizeof(int), cudaMemcpyHostToDevice);

    int *keys = new int[N] { 121,1,99,98,2,3,77,76,4,88,5,11,12,99,13,999,22,30,33,65,40,44,66 };
    int keys_count = 23;

    int *d_keys;
    cudaMalloc(&d_keys, keys_count*sizeof(int));
    cudaMemcpy(d_keys, keys, keys_count*sizeof(int), cudaMemcpyHostToDevice);

    search<<<4096, 256>>>(d_keys, keys_count, d_btree, N, degree, levels_count);

    cudaDeviceSynchronize();

    cudaFree(d_btree);
    
    delete [] btree;
}
