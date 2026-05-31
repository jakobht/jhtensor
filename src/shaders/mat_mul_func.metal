#include <metal_stdlib>
using namespace metal;

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
                     uint2 gid [[thread_position_in_grid]])
{

    uint row = gid.y;
    uint col = gid.x;

    T sum = 0;
    for (uint k = 0; k < params.K; k++) {
        sum += inA[row * params.K + k] * inB[k * params.N + col];
    }
    result[row * params.N + col] = sum;
}

kernel void mat_mul_f32(device const float* inA    [[buffer(0)]],
                           device const float* inB    [[buffer(1)]],
                           device float* result [[buffer(2)]],
                           constant MatMulDimensions& params [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]])
{
    mat_mul_impl<float>(inA, inB, result, params, gid);
}

kernel void mat_mul_i32(device const int* inA      [[buffer(0)]],
                           device const int* inB      [[buffer(1)]],
                           device int* result   [[buffer(2)]],
                           constant MatMulDimensions& params [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]])
{
    mat_mul_impl<int>(inA, inB, result, params, gid);
}

kernel void mat_mul_i16(device const short* inA     [[buffer(0)]],
                           device const short* inB     [[buffer(1)]],
                           device short* result  [[buffer(2)]],
                           constant MatMulDimensions& params [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]])
{
    mat_mul_impl<short>(inA, inB, result, params, gid);
}
