#include <torch/extension.h>
#include <vector>

// cuda forward declarations

torch::Tensor TOF_dist_proj_cuda_batchs(
    torch::Tensor image_batchs,
    torch::Tensor tof_value_batchs,
    torch::Tensor x1l_batchs, torch::Tensor y1l_batchs, torch::Tensor x1r_batchs, torch::Tensor y1r_batchs, 
    torch::Tensor x2l_batchs, torch::Tensor y2l_batchs, torch::Tensor x2r_batchs, torch::Tensor y2r_batchs,
    float time_resolution, const float dx, const float dy,
    const int nx, const int ny, const int event_num
);


torch::Tensor TOF_dist_bproj_cuda_batchs(
    torch::Tensor proj_batchs,
    torch::Tensor tof_value_batchs,
    torch::Tensor x1l_batchs, torch::Tensor y1l_batchs, torch::Tensor x1r_batchs, torch::Tensor y1r_batchs, 
    torch::Tensor x2l_batchs, torch::Tensor y2l_batchs, torch::Tensor x2r_batchs, torch::Tensor y2r_batchs,
    float time_resolution, const float dx, const float dy,
    const int nx, const int ny, const int event_num
);