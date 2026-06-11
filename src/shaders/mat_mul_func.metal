#include <metal_stdlib>
using namespace metal;

#define TILE_SIZE 16

struct MatMulDimensions {
    uint M; // The number of rows in the first matrix
    uint N; // The number of columns in the second matrix
    uint K; // The number of columns in the first matrix and the number of rows in the second matrix
};

template<typename T>
void mat_mul_impl(device const T* inA,
                     device const T* inB,
                     device T* result,
                     constant MatMulDimensions& params,
                     threadgroup T* tileA,
                     threadgroup T* tileB,
                     uint2 gid [[thread_position_in_grid]],
                     uint2 lid [[thread_position_in_threadgroup]])
{

    uint row = gid.y;
    uint col = gid.x;

    uint tile_row = lid.y;
    uint tile_col = lid.x;

    T sum = 0;

    for (uint tile_index = 0; tile_index < params.K; tile_index += TILE_SIZE) {
        // Load tiles
        if (row < params.M && tile_index + tile_col < params.K) {
            tileA[tile_row * TILE_SIZE + tile_col] = inA[row * params.K + tile_index + tile_col];
        } else {
            tileA[tile_row * TILE_SIZE + tile_col] = 0;
        }

        if (tile_index + tile_row < params.K && col < params.N) {
            tileB[tile_row * TILE_SIZE + tile_col] = inB[(tile_index + tile_row) * params.N + col];
        } else {
            tileB[tile_row * TILE_SIZE + tile_col] = 0;
        }

        // Wait for all tiles to be loaded
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Update the sum
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[tile_row * TILE_SIZE + k] * tileB[k * TILE_SIZE + tile_col];
        }

        // Wait for all computations to be done
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < params.M && col < params.N) {
        result[row * params.N + col] = sum;
    }
}

kernel void mat_mul_f32(device const float* inA    [[buffer(0)]],
                           device const float* inB    [[buffer(1)]],
                           device float* result [[buffer(2)]],
                           constant MatMulDimensions& params [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]],
                           uint2 lid [[thread_position_in_threadgroup]])
{
    threadgroup float tileA[TILE_SIZE * TILE_SIZE];
    threadgroup float tileB[TILE_SIZE * TILE_SIZE];


    mat_mul_impl<float>(inA, inB, result, params, tileA, tileB, gid, lid);
}

kernel void mat_mul_i32(device const int* inA      [[buffer(0)]],
                           device const int* inB      [[buffer(1)]],
                           device int* result   [[buffer(2)]],
                           constant MatMulDimensions& params [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]],
                           uint2 lid [[thread_position_in_threadgroup]])
{
    threadgroup int tileA[TILE_SIZE * TILE_SIZE];
    threadgroup int tileB[TILE_SIZE * TILE_SIZE];

    mat_mul_impl<int>(inA, inB, result, params, tileA, tileB, gid, lid);
}

kernel void mat_mul_i16(device const short* inA     [[buffer(0)]],
                           device const short* inB     [[buffer(1)]],
                           device short* result  [[buffer(2)]],
                           constant MatMulDimensions& params [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]],
                           uint2 lid [[thread_position_in_threadgroup]])
{
    threadgroup short tileA[TILE_SIZE * TILE_SIZE];
    threadgroup short tileB[TILE_SIZE * TILE_SIZE];

    mat_mul_impl<short>(inA, inB, result, params, tileA, tileB, gid, lid);
}
