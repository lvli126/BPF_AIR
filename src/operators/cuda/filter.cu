#include<stdio.h>
#include<math.h>

#define abs(x) (x > 0 ? x : -(x))
#define MAX(a,b) (a > b ? a : b)
#define MIN(a,b) (a < b ? a : b)
#define PI 3.1415926
#define GRIDDIM 32
#define BLOCKDIM 1024 //32*32


extern "C" void TOF_filter(float *filter_v, const int nx, const int ny, const float tof_sigma);


__device__ void TOF_filter_in_freq(float *filter_v, const int ix, const int iy, const int nx, const int ny, const float tof_sigma)
{
    const float nx2 = nx / 2;
    const float ny2 = ny / 2;
    

    float w_2 = ((ix - (nx2 - 0.5)) / nx2) * ((ix - (nx2 - 0.5)) / nx2) + ((iy - (ny2 - 0.5)) / ny2) * ((ix - (nx2 - 0.5)) / nx2);
    float temp_ = PI * PI *  tof_sigma * tof_sigma *  w_2;
    *filter_v = j0f(temp_) / expf(temp_);  
}



__global__ void  TOF_filter_in_freq_kernel(float *filter_v, const int nx, const float ny, const float tof_sigma)
{
    int step = blockDim.x * gridDim.x;
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nx * ny; idx += step) 
    {
        int ix = fmodf(idx, nx);
        int iy = (idx - ix) / nx;
        TOF_filter_in_freq(filter_v+idx, ix, iy, nx, ny, tof_sigma);
    }
}



void TOF_filter(float *filter_v, const int nx, const int ny, const float tof_sigma)
{
    float *filter_v_d;
    cudaMalloc(&filter_v_d, nx * ny * sizeof(float));
    TOF_filter_in_freq_kernel<<<GRIDDIM, BLOCKDIM>>>(filter_v_d, nx, ny, tof_sigma);
    cudaDeviceSynchronize();
    cudaMemcpy(filter_v, filter_v_d, nx * ny * sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(filter_v_d);
}

