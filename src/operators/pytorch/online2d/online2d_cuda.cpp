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

// #define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
// #define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
// #define CHECK_INPUT(x) \ //
//     CHECK_CUDA(x); \ //
//     CHECK_CONTIGUOUS(x)


torch::Tensor tof_project(
    torch::Tensor image_batchs,
    torch::Tensor tof_value_batchs,
    torch::Tensor x1l_batchs, torch::Tensor y1l_batchs, torch::Tensor x1r_batchs, torch::Tensor y1r_batchs, 
    torch::Tensor x2l_batchs, torch::Tensor y2l_batchs, torch::Tensor x2r_batchs, torch::Tensor y2r_batchs,
    float time_resolution, const float dx, const float dy,
    const int nx, const int ny, const int event_num
)
{
    return TOF_dist_proj_cuda_batchs(
        image_batchs, tof_value_batchs, x1l_batchs, y1l_batchs, x1r_batchs, y1r_batchs, 
        x2l_batchs, y2l_batchs, x2r_batchs, y2r_batchs, 
        time_resolution, dx, dy, nx, ny, event_num
    );
}


torch::Tensor tof_backproject(
    torch::Tensor proj_batchs,
    torch::Tensor tof_value_batchs,
    torch::Tensor x1l_batchs, torch::Tensor y1l_batchs, torch::Tensor x1r_batchs, torch::Tensor y1r_batchs, 
    torch::Tensor x2l_batchs, torch::Tensor y2l_batchs, torch::Tensor x2r_batchs, torch::Tensor y2r_batchs,
    float time_resolution, const float dx, const float dy,
    const int nx, const int ny, const int event_num
)
{
    return TOF_dist_bproj_cuda_batchs(
        proj_batchs, tof_value_batchs, x1l_batchs, y1l_batchs, x1r_batchs, y1r_batchs, 
        x2l_batchs, y2l_batchs, x2r_batchs, y2r_batchs, 
        time_resolution, dx, dy, nx, ny, event_num
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME,m)
{
    m.def("project", &tof_project, "TOF distance-driven projection");
    m.def("backproject", &tof_backproject, "TOF distance-driven backprojection");
}