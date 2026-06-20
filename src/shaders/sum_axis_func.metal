#include <metal_stdlib>
using namespace metal;

struct SumAxisParams {
    uint rows; // The number of rows in the matrix
    uint cols; // The number of columns in the matrix
    uint axis; // The axis to sum over
};

template<typename T>
void sum_axis_impl(device const T* in,
                     device T* result,
                     constant SumAxisParams& params,
                     uint gid [[thread_position_in_grid]])
{
    if (params.axis == 0 && gid < params.cols) {
        uint col = gid;

        T sum = 0;
        for (uint i = 0; i < params.rows; i++) {
            sum += in[i * params.cols + col];
        }
        result[col] = sum;
    } else if (params.axis == 1 && gid < params.rows) {
        uint row = gid;

        T sum = 0;
        for (uint j = 0; j < params.cols; j++) {
            sum += in[row * params.cols + j];
        }
        result[row] = sum;
    }
}

kernel void sum_axis_f32(device const float* in    [[buffer(0)]],
                           device float* result [[buffer(1)]],
                           constant SumAxisParams& params [[buffer(2)]],
                           uint gid [[thread_position_in_grid]])
{
    sum_axis_impl<float>(in, result, params, gid);
}

kernel void sum_axis_i32(device const int* in      [[buffer(0)]],
                           device int* result   [[buffer(1)]],
                           constant SumAxisParams& params [[buffer(2)]],
                           uint gid [[thread_position_in_grid]])
{
    sum_axis_impl<int>(in, result, params, gid);
}

kernel void sum_axis_i16(device const short* in     [[buffer(0)]],
                           device short* result  [[buffer(1)]],
                           constant SumAxisParams& params [[buffer(2)]],
                           uint gid [[thread_position_in_grid]])
{
    sum_axis_impl<short>(in, result, params, gid);
}
