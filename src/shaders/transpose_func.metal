#include <metal_stdlib>
using namespace metal;

struct TransposeParams {
    uint rows; // The number of rows in the matrix
    uint cols; // The number of columns in the matrix
};

template<typename T>
void transpose_impl(device const T* in,
                     device T* result,
                     constant TransposeParams& params,
                     uint2 gid [[thread_position_in_grid]],
                     uint2 lid [[thread_position_in_threadgroup]])
{

    uint row = gid.y;
    uint col = gid.x;

    if (row < params.rows && col < params.cols) {
        result[col * params.rows + row] = in[row * params.cols + col];
    }
}

kernel void transpose_f32(device const float* in    [[buffer(0)]],
                           device float* result [[buffer(1)]],
                           constant TransposeParams& params [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]],
                           uint2 lid [[thread_position_in_threadgroup]])
{
    transpose_impl<float>(in, result, params, gid, lid);
}

kernel void transpose_i32(device const int* in      [[buffer(0)]],
                           device int* result   [[buffer(1)]],
                           constant TransposeParams& params [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]],
                           uint2 lid [[thread_position_in_threadgroup]])
{
    transpose_impl<int>(in, result, params, gid, lid);
}

kernel void transpose_i16(device const short* in     [[buffer(0)]],
                           device short* result  [[buffer(1)]],
                           constant TransposeParams& params [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]],
                           uint2 lid [[thread_position_in_threadgroup]])
{
    transpose_impl<short>(in, result, params, gid, lid);
}
