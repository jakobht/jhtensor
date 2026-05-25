#include <metal_stdlib>
using namespace metal;

template<typename T>
void add_arrays_impl(device const T* inA,
                     device const T* inB,
                     device T* result,
                     uint            index)
{
    result[index] = inA[index] + inB[index];
}

kernel void add_arrays_f32(device const float* inA    [[buffer(0)]],
                           device const float* inB    [[buffer(1)]],
                           device float* result [[buffer(2)]],
                           uint                index  [[thread_position_in_grid]])
{
    add_arrays_impl<float>(inA, inB, result, index);
}

kernel void add_arrays_i32(device const int* inA      [[buffer(0)]],
                           device const int* inB      [[buffer(1)]],
                           device int* result   [[buffer(2)]],
                           uint              index    [[thread_position_in_grid]])
{
    add_arrays_impl<int>(inA, inB, result, index);
}

kernel void add_arrays_i16(device const short* inA     [[buffer(0)]],
                           device const short* inB     [[buffer(1)]],
                           device short* result  [[buffer(2)]],
                           uint                index   [[thread_position_in_grid]])
{
    add_arrays_impl<short>(inA, inB, result, index);
}
