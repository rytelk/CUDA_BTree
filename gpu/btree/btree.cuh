#pragma once

#include <iostream>
#include <stdio.h>
#include <chrono>
#include <cstring>
#include <ctime>

__host__ void gpu_btree_test()
{
    std::cout << "\nB+Tree GPU" << std::endl;    
    uint64_t *key;

    cudaEventCreate(&start);
    cudaEventRecord(start,0);

    btree_search<<<1, 4>>>(
        
    );

    cudaDeviceSynchronize();
  
    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&elapsedTime, start,stop);
    printf("Time elapsed: %f ms\n" ,elapsedTime);

    cudaFree(&found_key);
}