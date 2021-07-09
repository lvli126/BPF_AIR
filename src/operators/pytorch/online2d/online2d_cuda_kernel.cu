#include <cuda.h>
#include <torch/extension.h>

#include <cuda_runtime.h>
#include <ATen/ATen.h> // C++ front end的底层库，aten是一个tensor库，可以将数组封装成一个tensor类
#include <math.h>
#include <vector>


#define MAX(a,b) (a > b ? a : b)
#define MIN(a,b) (a < b ? a : b)
#define PI 3.1415926
// #define GRIDDIM 32
#define BLOCKDIM 1024 //32x*32y

texture<float> text_memory_image;
texture<float> text_memory_proj;

template <typename scalar_t> // scalar_t为tensor类变量在实际运行时的数据类型
__device__ void TOF_dist_proj(
    scalar_t *proj_value, 
    // const scalar_t *image, 
    const float tof_value,
    const float x1l, const float y1l, const float x1r, const float y1r,
    const float x2l, const float y2l, const float x2r, const float y2r,
    const float time_resolution, const float dx, const float dy,
    const int nx, const int n#define abs(x) (x > 0 ? x : -(x))

{
    const float nx2 = nx/2;
    const float ny2 = ny/2;
    const float tof_sigma = time_resolution * 0.3 / 2.355 / 2;
    const float tof_sigma_2 = tof_sigma * tof_sigma;
    const float x1c = (x1l + x1r) / 2;
    const float y1c = (y1l + y1r) / 2;
    const float x2c = (x2l + x2r) / 2;
    const float y2c = (y2l + y2r) / 2;
    const float L = sqrtf((x1c - x2c) *(x1c - x2c) + (y1c - y2c) * (y1c - y2c));
    const float ratio1 = (1 - tof_value / L) / 2;

    float d2_tof, w_tof;
    if (abs(x1c-x2c) > abs(y1c-y2c))
    {
        for (int ix=0; ix < nx; ix++)
        {
            float xc = (ix - nx2+0.5) * dx;
            float tof_bin = dx;
            
            
            if (tof_sigma > 0)
            {
                d2_tof = ((xc-x1c) / (x2c-x1c) - ratio1) * L;
                if (d2_tof <= (3 * tof_sigma))
                {
                    w_tof = expf(-0.5*d2_tof*d2_tof/tof_sigma_2)/sqrtf(2.0*PI*tof_sigma_2)*tof_bin;
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

            float yy1 = fminf(ylr,yrl); // 横坐标为xc时，detector边缘与x轴的交点中y较小值
            float yy2 = fmaxf(ylr,yrl);
            int cy1 = (int)floor(yy1/dy);
            int cy2 = (int)floor(yy2/dy);
            for (int iy = MAX(0, cy1); iy < MIN(ny, cy2+1); iy++)
            {
                float dist_w = (fminf((iy+1) * dy,yy2)-fmaxf(iy * dy, yy1)) / (yy2-yy1);
                atomicAdd(proj_value, text1Dfetch(text_memory_image, ix + nx * iy) * dist_w * w_tof);
            }
        }
    }
    else
    {
        //float kxlr, xlr, kxrl, xrl, 
        for (int iy=0; iy < ny; iy++)
        {
            float yc = (iy - ny2 + 0.5) *dy;
            float tof_bin = dy;
            if (tof_sigma > 0)
            {
                d2_tof = ((yc-y1c) / (y2c-y1c) - ratio1)*L;
                if (d2_tof <=3 * tof_sigma)
                {
                    w_tof = expf(-0.5 * d2_tof * d2_tof / tof_sigma_2) / sqrtf(2.0 * PI * tof_sigma_2) *tof_bin;
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
            
            float xx1 = fminf(xlr,xrl);
            float xx2 = fmaxf(xlr,xrl);
            float cx1 = (int)floor(xx1/dx);
            float cx2 = (int)floor(xx2/dx);

            
            for (int ix= MAX(0, cx1); ix < MIN(nx, cx2+1); ix++)
            {
                float dist_w = (fminf((ix+1)*dx,xx2) - fmaxf(ix*dx,xx1))/(xx2-xx1);
                atomicAdd(proj_value, text1Dfetch(text_memory_image, ix + nx * iy) * dist_w * w_tof);
            }
        }

    }
}
                


template <typename scalar_t>
__global__ void TOF_dist_proj_kernel(
    scalar_t *proj_value, 
    // const scalar_t *image, 
    const scalar_t *tof_value,
    const scalar_t *x1l, const scalar_t *y1l, const scalar_t *x1r, const scalar_t *y1r,
    const scalar_t *x2l, const scalar_t *y2l, const scalar_t *x2r, const scalar_t *y2r,
    const float time_resolution, const float dx, const float dy,
    const int nx, const int ny, const int event_num)
{
    int step = blockDim.x * gridDim.x;
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < event_num; idx += step)
    {
        TOF_dist_proj(proj_value+idx, tof_value[idx], x1l[idx], y1l[idx], x1r[idx], y1r[idx],
                    x2l[idx], y2l[idx], x2r[idx], y2r[idx], time_resolution, dx, dy, nx, ny);
    }
}


torch::Tensor TOF_dist_proj_cuda_batchs(
    torch::Tensor image_batchs,
    torch::Tensor tof_value_batchs,
    torch::Tensor x1l_batchs, torch::Tensor y1l_batchs, torch::Tensor x1r_batchs, torch::Tensor y1r_batchs, 
    torch::Tensor x2l_batchs, torch::Tensor y2l_batchs, torch::Tensor x2r_batchs, torch::Tensor y2r_batchs,
    float time_resolution, const float dx, const float dy,
    const int nx, const int ny, const int event_num) 
{
    const int nb_batchs = image_batchs.size(0);
    const int nb_channels = image_batchs.size(1);
    torch::Tensor proj_batchs = torch::zeros({nb_batchs, nb_channels, event_num, 1}, image_batchs.type());
    

    for (int ibatch = 0; ibatch < nb_batchs; ibatch++)
    {
        cudaBindTexture(0, text_memory_image, image_batchs.data<scalar_t>()+ibatch*nx*ny,
                    nx*ny*sizeof(float)); //将image绑定在纹理内存中，只读，偏移量为0
        dim3 dimBlock = BLOCKDIM;
        dim3 dimGrid = (event_num -1)/dimBlock + 1;

        AT_DISPATCH_FLOATING_TYPES(
            at::ScalarType::Float,
            "TOF_dist_proj_cuda_batchs",
            ([&] {
                    TOF_dist_proj_kernel<scalar_t><<<dimGrid, dimBlock>>>(
                        proj_batchs.data<scalar_t>()+ibatch*event_num*1,
                        //image_batchs.data<scalar_t>(),
                        tof_value_batchs.data<scalar_t>()+ibatch*event_num*1,
                        x1l_batchs.data<scalar_t>()+ibatch*event_num*1,
                        y1l_batchs.data<scalar_t>()+ibatch*event_num*1,
                        x1r_batchs.data<scalar_t>()+ibatch*event_num*1,
                        y1r_batchs.data<scalar_t>()+ibatch*event_num*1,
                        x2l_batchs.data<scalar_t>()+ibatch*event_num*1,
                        y2l_batchs.data<scalar_t>()+ibatch*event_num*1,
                        x2r_batchs.data<scalar_t>()+ibatch*event_num*1,
                        y2r_batchs.data<scalar_t>()+ibatch*event_num*1,
                        time_resolution,
                        dx,dy,nx,ny,event_num);
            }));
        cudaDeviceSynchronize();
        cudaUnbindTexture(text_memory_image);
            
    }
    return proj_batchs;
}


template <scalar_t>
__device__ void TOF_dist_bp(
    scalar_t *image_bp, 
    const float proj_value,
    const float tof_value,
    const float x1l, const float y1l, const float x1r, const float y1r,
    const float x2l, const float y2l, const float x2r, const float y2r,
    const float time_resolution, const float dx, const float dy,
    const int nx, const int ny)
{
    const float nx2 = nx/2;
    const float ny2 = ny/2;
    const float tof_sigma = time_resolution * 0.3 / 2.355 / 2;
    const float tof_sigma_2 = tof_sigma * tof_sigma;
    const float x1c = (x1l + x1r) / 2;
    const float y1c = (y1l + y1r) / 2;
    const float x2c = (x2l + x2r) / 2;
    const float y2c = (y2l + y2r) / 2;
    const float L = sqrtf((x1c - x2c) * (x1c - x2c) + (y1c - y2c) * (y1c - y2c));
    const float ratio1 = (1 - (tof_value / L)) / 2;

    
    
    if (abs(x1c - x2c) > abs(y1c - y2c))
    {
        for (int ix = 0; ix < nx; ix++)
        {
            float xc = (ix - nx2 + 0.5) * dx;
            float tof_bin = dx;
            float d2_tof, w_tof;
            
            if (tof_sigma > 0)
            {
                d2_tof = ((xc-x1c) / (x2c-x1c) - ratio1)*L;
                if (d2_tof <= 3 * tof_sigma)
                {
                    w_tof = expf(-0.5 * d2_tof * d2_tof / tof_sigma_2) / sqrtf(2.0 * PI * tof_sigma_2) * tof_bin;
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
            float ylr = kylr * (xc - x1l) + y1l + ny2 * dy;
            //d1r-d2l
            float kyrl = (y1r - y2l) / (x1r - x2l);
            float yrl = kyrl * (xc - x1r) + y1r + ny2 * dy;

            float yy1 = MIN(ylr,yrl); // 横坐标为xc时，detector边缘与x轴的交点中y较小值
            float yy2 = MAX(ylr,yrl);
            int cy1 = (int)floorf(yy1/dy);
            int cy2 = (int)floorf(yy2/dy);

            for (int iy=(int)MAX(0, cy1); iy < (int)MIN(ny, cy2+1); iy++)
            {
                float dist_w = (MIN((iy+1) * dy,yy2) - MAX(iy * dy,yy1)) / dy;
                atomicAdd(image_bp + (ix + iy * nx), text_memory_proj * dist_w * w_tof);
            }

        }
        
    }
    else
    {
        for (int iy=0; iy < ny; iy++)
        {
            float yc = (iy - ny2 + 0.5) * dy;
            float tof_bin = dy;
            float d2_tof, w_tof;

            if (tof_sigma > 0)
            {
                d2_tof = (((yc-y1c) / (y2c-y1c)) - ratio1) * L;
                if (d2_tof <= 3 * tof_sigma)
                {
                    w_tof = expf(-0.5 * d2_tof * d2_tof / tof_sigma_2) / sqrtf(2.0 * PI * tof_sigma_2) * tof_bin;
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
            float xlr = kxlr * (yc-y1l)+x1l+nx2 * dx;
            //d1r-d2l:
            float kxrl = (x1r-x2l)/(y1r-y2l);
            float xrl = kxrl * (yc-y1r)+x1r+nx2 * dx;
            
            float xx1 = MIN(xlr,xrl);
            float xx2 = MAX(xlr,xrl);
            float cx1 = (int)floorf(xx1/dx);
            float cx2 = (int)floorf(xx2/dx);

            
            for (int ix=(int)MAX(0, cx1); ix < (int)MIN(nx, cx2+1); ix++)
            {
                float dist_w = (MIN((ix+1) * dx,xx2) - MAX(ix * dx,xx1))/dx;
                atomicAdd(image_bp + (ix + iy * nx), proj_value * dist_w * w_tof);
            }

        }
    }

}


template <scalar_t>
__global__ void TOF_dist_bp_kernel(
    scalar_t *image_bp, 
    // const scalar_t *proj_value, 
    const scalar_t *tof_value,                               
    const scalar_t *x1l, const scalar_t *y1l, const scalar_t *x1r, const scalar_t *y1r,
    const scalar_t *x2l, const scalar_t *y2l, const scalar_t *x2r, const scalar_t *y2r,
    const float time_resolution, const float dx, const float dy,
    const int nx, const int ny, const int event_num)
{
    int step = blockDim.x * gridDim.x;
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < event_num; idx += step)
    {
        TOF_dist_bp(image_bp, text1Dfetch(text_memory_proj,idx), tof_value[idx],
                    x1l[idx], y1l[idx], x1r[idx], y1r[idx],
                    x2l[idx], y2l[idx], x2r[idx], y2r[idx],
                    time_resolution, dx, dy,
                    nx, ny);
    }

}

torch::Tensor TOF_dist_bproj_cuda_batchs(
    torch::Tensor proj_batchs,
    torch::Tensor tof_value_batchs,
    torch::Tensor x1l_batchs, torch::Tensor y1l_batchs, torch::Tensor x1r_batchs, torch::Tensor y1r_batchs, 
    torch::Tensor x2l_batchs, torch::Tensor y2l_batchs, torch::Tensor x2r_batchs, torch::Tensor y2r_batchs,
    float time_resolution, const float dx, const float dy,
    const int nx, const int ny, const int event_num) 
{
    const int nb_batchs = proj_batchs.size(0);
    const int nb_channels = proj_batchs.size(1);
    torch::Tensor back_image_batchs = torch::zeros({nb_batchs, nb_channels, nx, ny}, proj_batchs.type());

    dim3 dimBlock = BLOCKDIM;
    dim3 dimGrid = (nx * ny -1) / dimBlock +1;

    for (int ibatch = 0; ibatch < nb_batchs; ibatch++)
    {
        cudaBindTexture(0, text_memory_proj, proj_batchs.data<scalar_t>() + ibatch * event_num *1,
                        event_num * sizeof(float));
        AT_DISPATCH_FLOATING_TYPES(
            at::ScalarType::Float,
            "TOF_dist_bproj_cuda_batchs",
            ([&]{
                TOF_dist_bp_kernel<scalar_t><<<dimGrid, dimBlock>>>(
                    back_image_batchs.data<scalar_t>()+ibatch*nx*ny,
                    tof_value_batchs.data<scalar_t>()+ibatch*nx*ny,
                    x1l_batchs.data<scalar_t>()+ibatch*nx*ny,
                    y1l_batchs.data<scalar_t>()+ibatch*nx*ny,
                    x1r_batchs.data<scalar_t>()+ibatch*nx*ny,
                    y1r_batchs.data<scalar_t>()+ibatch*nx*ny,
                    x2l_batchs.data<scalar_t>()+ibatch*nx*ny,
                    y2l_batchs.data<scalar_t>()+ibatch*nx*ny,
                    x2r_batchs.data<scalar_t>()+ibatch*nx*ny,
                    y2r_batchs.data<scalar_t>()+ibatch*nx*ny,
                    time_resolution,
                    dx,dy,nx,ny,event_num)

            });
        cudaDeviceSynchronize();
        cudaUnbindTexture(text_memory_proj);
    }
    return back_image_batchs;
}




// torch::Tensor bpf_batchs(
//     torch::Tensor proj_batchs,
//     torch::Tensor tof_value_batchs,
//     torch::Tensor x1l_batchs, torch::Tensor y1l_batchs, torch::Tensor x1r_batchs, torch::Tensor y1r_batchs, 
//     torch::Tensor x2l_batchs, torch::Tensor y2l_batchs, torch::Tensor x2r_batchs, torch::Tensor y2r_batchs,
//     float time_resolution, const float dx, const float dy,
//     const int nx, const int ny, const int event_num) 
// {


//     back_image_batchs = TOF_dist_bproj_cuda_batchs(proj_batchs, tof_value_batchs, 
//                                                    x1l_batchs, y1l_batchs, x1r_batchs,y1r_batchs,
//                                                    x1l_batchs, y1l_batchs, x1r_batchs,y1r_batchs,
//                                                    time_resolution, 
//                                                    dx,dy,nx,ny,event_num)
    
// }