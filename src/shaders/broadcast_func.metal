#include <metal_stdlib>
using namespace metal;

#define TILE_SIZE 16

struct BroadcastParams {
    uint rows; // The number of rows in the matrix
    uint cols; // The number of columns in the matrix
    uint axis; // The axis to broadcast over
};

template<typename T>
void broadcast_impl(device const T* in,
                     device T* result,
                     constant BroadcastParams& params,
                     uint2 gid [[thread_position_in_grid]],
                     uint2 lid [[thread_position_in_threadgroup]])
{

    uint row = gid.y;
    uint col = gid.x;

    if (row < params.rows && col < params.cols) {
        if (params.axis == 0) {
            result[row * params.cols + col] = in[col];
        } else if (params.axis == 1) {
            result[row * params.cols + col] = in[row];
        }
    }
}

kernel void broadcast_f32(device const float* in    [[buffer(0)]],
                           device float* result [[buffer(1)]],
                           constant BroadcastParams& params [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]],
                           uint2 lid [[thread_position_in_threadgroup]])
{
    broadcast_impl<float>(in, result, params, gid, lid);
}

kernel void broadcast_i32(device const int* in      [[buffer(0)]],
                           device int* result   [[buffer(1)]],
                           constant BroadcastParams& params [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]],
                           uint2 lid [[thread_position_in_threadgroup]])
{
    broadcast_impl<int>(in, result, params, gid, lid);
}

kernel void broadcast_i16(device const short* in     [[buffer(0)]],
                           device short* result  [[buffer(1)]],
                           constant BroadcastParams& params [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]],
                           uint2 lid [[thread_position_in_threadgroup]])
{
    broadcast_impl<short>(in, result, params, gid, lid);
}
