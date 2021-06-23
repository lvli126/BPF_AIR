#include<stdio.h>
#include<math.h>

#define abs(x) (x > 0 ? x : -(x))
#define MAX(a,b) (a > b ? a : b)
#define MIN(a,b) (a < b ? a : b)
#define PI 3.1415926
#define GRIDDIM 32
#define BLOCKDIM 1024 //32*32


extern "C" void TOF_filter(float *filter_v, const float nx, const int ny, const float time_resolution);


__device__ void TOF_filter_in_freq(float *filter_v, const int ix, const int iy, const float nx, const float ny, const float time_resolution)
{
    const float nx2 = nx / 2;
    const float ny2 = ny / 2;
    const float tof_sigma = time_resolution * 0.3 / 2.355 / 2;

    float w_2 = ((ix - (nx2 - 0.5)) / nx2) * ((ix - (nx2 - 0.5)) / nx2) + ((iy - (ny2 - 0.5)) / ny2) * ((ix - (nx2 - 0.5)) / nx2);
    float temp_ = PI * PI *  tof_sigma * tof_sigma *  w_2;
    *filter_v = j0f(temp_) / expf(temp_);  
}



__global__ void  TOF_filter_in_freq_kernel(float *filter_v, const float nx, const float ny, const float time_resolution)
{
    int step = blockDim.x * gridDim.x;
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nx * ny; idx += step) 
    {
        int ix = fmodf(idx, nx);
        int iy = (idx - ix) / nx;
        TOF_filter_in_freq(filter_v+idx, ix, iy, nx, ny, time_resolution);
    }
}



void TOF_filter(float *filter_v, const float nx, const int ny, const float time_resolution)
{
    float *filter_v_d;
    cudaMalloc(&filter_v_d, nx * ny * sizeof(float));
    TOF_filter_in_freq_kernel<<<GRIDDIM, BLOCKDIM>>>(filter_v_d, nx, ny, time_resolution);
    cudaDeviceSynchronize();
    cudaMemcpy(filter_v, filter_v_d, nx * ny * sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(filter_v_d);
}

