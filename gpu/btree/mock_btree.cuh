#include <iostream>

__host__
int* get_mock_btree0(int N, int &degree, int &levels_count) ;

__host__
int* get_mock_btree1(int N, int &degree, int &levels_count) ;

__host__
int* get_mock_btree2(int N, int &degree, int &levels_count) ;

__host__
int* get_mock_btree0(int N, int &degree, int &levels_count) 
{
    degree = 4;
    levels_count = 0;
    return new int[N] { 3, 5, 12 };
}

__host__
int* get_mock_btree1(int N, int &degree, int &levels_count)
{
    degree = 4;
    levels_count = 1;
    return new int[N] { 5, 13, 30, 1, 2, 4, 5, 12, -1, 13, 22, -1, 30, 33, -1 };
}

__host__
int* get_mock_btree2(int N, int &degree, int &levels_count)
{
    degree = 4;
    levels_count = 2;
    return new int[N] { 13,-1,-1,-1,3,5,-1,-1,30,40,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,2,-1,-1,3,4,-1,-1,5,11,12,-1,-1,-1,-1,-1,13,22,-1,-1,30,33,-1,-1,40,44,-1,-1 };
}
