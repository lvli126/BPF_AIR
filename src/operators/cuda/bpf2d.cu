#include <studio.h>
#include <math.h>

#define abs(x) (x > 0 ? x : -(x))
//#define max(a,b) (a > b ? a : b)
//#define min(a,b) (a < b ? a : b)
#define PI 3.1415926;
#define GRIDDIM 32
#define BLOCKDIM 1024 //32*32

extern "C" void TOF_dist_projection(float *proj_value, const float *image, const float *tof_value,
                            const float *x1l, const float *y1l, const float *x1r, const float *y1r,
                            const float *x2l, const float *y2l, const float *x2r, const float *y2r,
                            const float time_resolution, const float dx, const float dy,
                            const int nx, const int ny, const int event_num)

extern "C" void TOF_dist_backprojection(float *image_bp, const float *proj_value, const float *tof_value,
                            const float *x1l, const float *y1l, const float *x1r, const float *y1r,
                            const float *x2l, const float *y2l, const float *x2r, const float *y2r,
                            const float time_resolution, const float dx, const float dy,
                            const int nx, const int ny, const int event_num)

extern "C" void TOF_filter(float *filter_v, const float nx, const ny, const float time_resolution)

// define TOF-dist-proj
__device__ void TOF_dist_proj(float proj_value, const float *image, const float tof_value,
                              const float x1l, const float y1l, const float x1r, const float y1r,
                              const float x2l, const float y2l, const float x2r, const float y2r,
                              const float time_resolution, const float dx, const float dy,
                              const int nx, const int ny)
{
    const float nx2 = nx/2;
    const float ny2 = ny/2;
    const float tof_sigma = time_resolution * 0.3 / 2.355 / 2;
    const float tof_sigma2 = tof_sigma / 2;
    const float x1c = (x1l + x1r) / 2;
    const float y1c = (y1l + y1r) / 2;
    const float x2c = (x2l + x2r) / 2;
    const float y2c = (y2l + y2r) / 2;
    const float L = ((x1c - x2c) ** 2 + (y1c - y2c) ** 2) ** 0.5;
    const float ratio1 = (1 - tof_value / L) / 2;
    if (abs(x1c-x2c) > abs(y1c-y2c))
    {
        for (ix=0; ix < nx; ix++)
        {
            float xc = (ix - nx2+0.5) * dx;
            float tof_bin = dx;
            
            if (tof_sigma > 0)
            {
                d2_tof = ((xc-x1c) / (x2c-x1c) - ratio1)*L;
                if (d2_tof <=3 * tof_sigma)
                {
                    w_tof = (expf(-0.5 * d2_tof * d2_tof / tof_sigma2) / sqrtf(2.0 * PI * tof_sigma2)) *tof_bin;
                }
                else
                {
                    w_tof = 0.0;
                }
            }
            else
            {
                w_tof = 1.0;
            }
            //d1l-d2r
            float kylr = (y1l-y2r)/(x1l-x2r);
            float ylr = kylr*(xc-x1l)+y1l+ny2*dy;
            //d1r-d2l
            float kyrl = (y1r-y2l)/(x1r-x2l);
            float yrl = kyrl*(xc-x1r)+y1r+ny2*dy;

            float yy1 = min(ylr,yrl); // 横坐标为xc时，detector边缘与x轴的交点中y较小值
            float yy2 = max(ylr,yrl);
            int cy1 = floor(yy1/dy);
            int cy2 = floor(yy2/dy);
            for (int iy = max(0, cy1); iy < min(ny, cy2+1); iy++)
            {
                float dist_w = (min((iy+1)*dy,yy2)-max(iy*dy,yy1))/(yy2-yy1);
                atomicAdd(proj_value, (image_bp + (ix + nx * iy)) * dist_w * w_tof);
            }
        }
    }
    else
    {
        for (int iy=0; iy < ny; iy++)
        {
            float yc = (iy - ny2 + 0.5) *dy;
            float tof_bin = dy;
            if (tof_sigma > 0)
            {
                d2_tof = ((yc-y1c) / (y2c-y1c) - ratio1)*L;
                if (d2_tof <=3 * tof_sigma)
                {
                    w_tof = expf(-0.5 * d2_tof * d2_tof / tof_sigma2) / sqrtf(2.0 * PI * tof_sigma2) *tof_bin;
                }
                else
                {
                    w_tof = 0.0;
                }
            }
            else
            {
                w_tof = 1.0;
            }
            //d1l-d2r:
            float kxlr = (x1l-x2r)/(y1l-y2r);
            float xlr = kxlr*(yc-y1l)+x1l+nx2*dx;
            //d1r-d2l:
            float kxrl = (x1r-x2l)/(y1r-y2l);
            float xrl = kxrl*(yc-y1r)+x1r+nx2*dx;
            
            float xx1 = min(xlr,xrl);
            float xx2 = max(xlr,xrl);
            float cx1 = floor(xx1/dx);
            float cx2 = floor(xx2/dx);

            
            for (int ix=max(0, cx1); ix < min(nx, cx2+1); ix++)
            {
                float dist_w = (min((ix+1)*dx,xx2) - max(ix*dx,xx1))/(xx2-xx1);
                atomicAdd(proj_value, (image_bp + (ix + nx * iy)) * dist_w * w_tof);
            }

    }
}

// define TOF-dist-bp
__device__ void TOF_dist_bp(float *image_bp, const float proj_value, const float tof_value,
                            const float x1l, const float y1l, const float x1r, const float y1r,
                            const float x2l, const float y2l, const float x2r, const float y2r,
                            const float time_resolution, const float dx, const float dy,
                            const int nx, const int ny)
{
    const float nx2 = nx/2;
    const float ny2 = ny/2;
    const float tof_sigma = time_resolution * 0.3 / 2.355 / 2;
    const float tof_sigma2 = tof_sigma / 2;
    const float x1c = (x1l + x1r) / 2;
    const float y1c = (y1l + y1r) / 2;
    const float x2c = (x2l + x2r) / 2;
    const float y2c = (y2l + y2r) / 2;
    const float L = ((x1c - x2c) ** 2 + (y1c - y2c) ** 2) ** 0.5;
    const float ratio1 = (1 - tof_value / L) / 2;

    const float xd = x1c - x2c;
    const float yd = y1c - y2c;
    float d2_tof, w_tof;
    if (abs(xd) > abs(yd))
    {
        for (int ix = 0; ix < nx; ix++)
        {
            float xc = (ix - nx2+0.5) * dx;
            float tof_bin = dx;
            
            if (tof_sigma > 0)
            {
                d2_tof = ((xc-x1c) / (x2c-x1c) - ratio1)*L;
                if (d2_tof <=3 * tof_sigma)
                {
                    w_tof = expf(-0.5 * d2_tof * d2_tof / tof_sigma2) / sqrtf(2.0 * PI * tof_sigma2) *tof_bin;
                }
                else
                {
                    w_tof = 0.0;
                }
            }
            else
            {
                w_tof = 1.0;
            }
            //d1l-d2r
            float kylr = (y1l-y2r)/(x1l-x2r);
            float ylr = kylr*(xc-x1l)+y1l+ny2*dy;
            //d1r-d2l
            float kyrl = (y1r-y2l)/(x1r-x2l);
            float yrl = kyrl*(xc-x1r)+y1r+ny2*dy;

            float yy1 = min(ylr,yrl); // 横坐标为xc时，detector边缘与x轴的交点中y较小值
            float yy2 = max(ylr,yrl);
            int cy1 = floor(yy1/dy);
            int cy2 = floor(yy2/dy);

            for (int iy=max(0, cy1); iy < min(ny, cy2+1); iy++)
            {
                float dist_w = (min((iy+1)*dy,yy2)-max(iy*dy,yy1))/dy;
                atomicAdd(image + (ix + iy * nx), proj_value * dist_w * w_tof);
            }

        }
        
    }
    else
    {
        for (int iy=0; iy < ny; iy++)
        {
            float yc = (iy - ny2 + 0.5) *dy;
            float tof_bin = dy;
            if (tof_sigma > 0)
            {
                d2_tof = ((yc-y1c) / (y2c-y1c) - ratio1)*L;
                if (d2_tof <=3 * tof_sigma)
                {
                    w_tof = expf(-0.5 * d2_tof * d2_tof / tof_sigma2) / sqrtf(2.0 * PI * tof_sigma2) *tof_bin;
                }
                else
                {
                    w_tof = 0.0;
                }
            }
            else
            {
                w_tof = 1.0;
            }
            //d1l-d2r:
            float kxlr = (x1l-x2r)/(y1l-y2r);
            float xlr = kxlr*(yc-y1l)+x1l+nx2*dx;
            //d1r-d2l:
            float kxrl = (x1r-x2l)/(y1r-y2l);
            float xrl = kxrl*(yc-y1r)+x1r+nx2*dx;
            
            float xx1 = min(xlr,xrl);
            float xx2 = max(xlr,xrl);
            float cx1 = floor(xx1/dx);
            float cx2 = floor(xx2/dx);

            
            for (int ix=max(0, cx1); ix < min(nx, cx2+1); ix++)
            {
                float dist_w = (min((ix+1)*dx,xx2) - max(ix*dx,xx1))/dx;
                atomicAdd(image + (ix + iy * nx), proj_value * dist_w * w_tof);
            }

        }
    }

}

__device__ void TOF_filter_in_freq(float *filter_v, const int ix, const int iy, const float nx, const float ny, const float time_resolution)
{
    const float nx2 = nx / 2;
    const float ny2 = ny / 2;
    const float tof_sigma = time_resolution * 0.3 / 2.355 / 2;

    float w_2 = ((ix - (nx2 - 0.5)) / nx2) * ((ix - (nx2 - 0.5)) / nx2) + ((iy - (ny2 - 0.5)) / ny2) * ((ix - (nx2 - 0.5)) / nx2);
    float temp_ = PI * PI *  tof_sigma * tof_sigma *  w_2;
    filter_v = j0f(temp_) / expf(temp_);  
}


__global__ void TOF_dist_proj_kernel(float *proj_value, const float *image, const float *tof_value,
                                    const float *x1l, const float *y1l, const float *x1r, const float *y1r,
                                    const float *x2l, const float *y2l, const float *x2r, const float *y2r,
                                    const float time_resolution, const float dx, const float dy,
                                    const int nx, const int ny, const int event_num)
{
    int step = blockDim.x * gridDim.x;
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < event_num; idx += step)
    {
        TOF_dist_proj(proj_value[idx], image, tof_value[idx], x1l[idx], y1l[idx], x1r[idx], y1r[idx],
                    x2l[idx], y2l[idx], x2r[idx], y2r[idx], time_resolution, dx, dy, nx, ny);
    }
}


__global__ void TOF_dist_bp_kernel(float *image_bp, const float *proj_value, const float *tof_value,
                                   const float *x1l, const float *y1l, const float *x1r, const float *y1r,
                                   const float *x2l, const float *y2l, const float *x2r, const float *y2r,
                                   const float time_resolution, const float dx, const float dy,
                                   const int nx, const int ny, const int event_num)
{
    int step = blockDim.x * gridDim.x;
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < event_num; idx += step)
    {
        TOF_dist_bp(image_bp, proj_value[idx], tof_value[idx],
                    x1l[idx], y1l[idx], x1r[idx], y1r[idx],
                    x2l[idx], y2l[idx], x2r[idx], y2r[idx],
                    time_resolution, dx, dy,
                    nx, ny);
    }

}


__global__ void  TOF_filter_in_freq_kernel(float *filter_v, const float nx, const float ny, const float time_resolution)
{
    int step = blockDim.x * gridDim.x
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nx * ny; idx += step) 
    {
        int ix = fmodf(idx, nx);
        int iy = (idx - ix) / nx;
        TOF_filter_in_freq(filter_v[idx], ix, iy, nx, ny, time_resolution);
    }
}


void TOF_dist_projection(float *proj_value, const float *image, const float *tof_value,
                            const float *x1l, const float *y1l, const float *x1r, const float *y1r,
                            const float *x2l, const float *y2l, const float *x2r, const float *y2r,
                            const float time_resolution, const float dx, const float dy,
                            const int nx, const int ny, const int event_num)
{
    float *image_d, *proj_value_d, *tof_value_d, *x1l_d, *y1l_d, *x1r_d, *y1r_d, *x2l_d, *y2l_d, *x2r_d, *y2r_d;
    // 申请device内存
    cudaMalloc(&image_d, nx * ny * sizeof(float));
    cudaMalloc(&proj_value_d, event_num * sizeof(float));
    cudaMalloc(&tof_value_d, event_num * sizeof(float));
    cudaMalloc(&x1l_d, event_num * sizeof(float));
    cudaMalloc(&y1l_d, event_num * sizeof(float));
    cudaMalloc(&x1r_d, event_num * sizeof(float));
    cudaMalloc(&y1r_d, event_num * sizeof(float));
    cudaMalloc(&x2l_d, event_num * sizeof(float));
    cudaMalloc(&y2l_d, event_num * sizeof(float));
    cudaMalloc(&x2r_d, event_num * sizeof(float));
    cudaMalloc(&y2r_d, event_num * sizeof(float));
    // 将host上的数据拷贝到device上
    cudaMemcpy(image_d, image, event_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(tof_value_d, tof_value, event_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x1l_d, x1l, event_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y1l_d, y1l, event_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x1r_d, x1r, event_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y1r_d, x1r, event_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x2l_d, x2l, event_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y2l_d, y2l, event_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x2r_d, x2r, event_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y2r_d, x2r, event_num * sizeof(float), cudaMemcpyHostToDevice);
    // 执行kernel
    TOF_dist_proj_kernel<<<GRIDDIM, BLOCKDIM>>>(proj_value_d, image_d, tof_value_d, x1l_d, y1l_d, x1r_d, y1r_d,
                                            x2l_d, y2l_d, x2r_d, y2r_d, time_resolution, dx, dy, nx, ny, event_num);
    // 同步device，保证结果正常访问
    cudaDeviceSynchronize();
    // 将device结果cpy到host
    cudaMemcpy(proj_value, proj_value_d, nx * ny * sizeof(float), cudaMemcpyDeviceToHost);
    // 释放内存
    cudaFree(image_d);
    cudaFree(proj_value_d);
    cudaFree(tof_value_d);
    cudaFree(x1l_d);
    cudaFree(y1l_d);
    cudaFree(x1r_d);
    cudaFree(y1r_d);
    cudaFree(x2l_d);
    cudaFree(y2l_d);
    cudaFree(x2r_d);
    cudaFree(y2r_d);
}

void TOF_dist_backprojection(float *image_bp, const float *proj_value, const float *tof_value,
                            const float *x1l, const float *y1l, const float *x1r, const float *y1r,
                            const float *x2l, const float *y2l, const float *x2r, const float *y2r,
                            const float time_resolution, const float dx, const float dy,
                            const int nx, const int ny, const int event_num)
{
    float *image_bp_d, *proj_value_d, *tof_value_d, *x1l_d, *y1l_d, *x1r_d, *y1r_d, *x2l_d, *y2l_d, *x2r_d, *y2r_d;
    // 申请device内存
    cudaMalloc(&image_bp_d, nx * ny * sizeof(float));
    cudaMalloc(&proj_value_d, event_num * sizeof(float));
    cudaMalloc(&tof_value_d, event_num * sizeof(float));
    cudaMalloc(&x1l_d, event_num * sizeof(float));
    cudaMalloc(&y1l_d, event_num * sizeof(float));
    cudaMalloc(&x1r_d, event_num * sizeof(float));
    cudaMalloc(&y1r_d, event_num * sizeof(float));
    cudaMalloc(&x2l_d, event_num * sizeof(float));
    cudaMalloc(&y2l_d, event_num * sizeof(float));
    cudaMalloc(&x2r_d, event_num * sizeof(float));
    cudaMalloc(&y2r_d, event_num * sizeof(float));
    // 将host上的数据拷贝到device上
    cudaMemcpy(proj_value_d, proj_value, event_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(tof_value_d, tof_value, event_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x1l_d, x1l, event_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y1l_d, y1l, event_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x1r_d, x1r, event_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y1r_d, x1r, event_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x2l_d, x2l, event_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y2l_d, y2l, event_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x2r_d, x2r, event_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y2r_d, x2r, event_num * sizeof(float), cudaMemcpyHostToDevice);
    // 执行kernel
    TOF_dist_bp_kernel<<<GRIDDIM, BLOCKDIM>>>(image_bp_d, proj_value_d, tof_value_d, x1l_d, y1l_d, x1r_d, y1r_d,
                                            x2l_d, y2l_d, x2r_d, y2r_d, time_resolution, dx, dy, nx, ny, event_num);
    // 同步device，保证结果正常访问
    cudaDeviceSynchronize();
    // 将device结果cpy到host
    cudaMemcpy(image_bp, image_bp_d, nx * ny * sizeof(float), cudaMemcpyDeviceToHost);
    // 释放内存
    cudaFree(image_bp_d);
    cudaFree(proj_value_d);
    cudaFree(tof_value_d);
    cudaFree(x1l_d);
    cudaFree(y1l_d);
    cudaFree(x1r_d);
    cudaFree(y1r_d);
    cudaFree(x2l_d);
    cudaFree(y2l_d);
    cudaFree(x2r_d);
    cudaFree(y2r_d);
}


void TOF_filter(float *filter_v, const float nx, const ny, const float time_resolution)
{
    float *filter_v_d;
    cudaMalloc(&filter_v_d, nx * ny * sizeof(float);)
    TOF_filter_in_freq_kernel<<<GRIDDIM, BLOCKDIM>>>(filter_v_d, nx, ny, time_resolution);
    cudaDeviceSynchronize();
    cudaMemcpy(filter_v, filter_v_d, nx * ny * sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(filter_v_d);
}

