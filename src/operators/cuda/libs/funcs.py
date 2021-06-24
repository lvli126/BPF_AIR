import ctypes
from ctypes import *

def get_TOF_dist_backprojection():
    dll = ctypes.CDLL('./libs/bproj.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.TOF_dist_backprojection
    func.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float),
                     POINTER(c_float), POINTER(c_float), POINTER(c_float),POINTER(c_float),
                     POINTER(c_float), POINTER(c_float), POINTER(c_float),POINTER(c_float),
                     c_float, c_float, c_float, 
                     c_uint32, c_uint32, c_uint32]
    return func

def get_TOF_dist_projection():
    dll = ctypes.CDLL("./libs/proj.so", mode=ctypes.RTLD_GLOBAL)
    func = dll.TOF_dist_projection
    func.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float),
                     POINTER(c_float), POINTER(c_float), POINTER(c_float),POINTER(c_float),
                     POINTER(c_float), POINTER(c_float), POINTER(c_float),POINTER(c_float),
                     c_float, c_float, c_float, 
                     c_uint32, c_uint32, c_uint32] 
    return func

def get_TOF_filter():
    dll = ctypes.CDLL("./libs/filter.so", mode=ctypes.RTLD_GLOBAL)
    func = dll.TOF_filter
    func.argtypes = [POINTER(c_float), c_uint32, c_uint32, c_float]
    return func

# TOF_dist_projection(float *proj_value, const float *image, const float *tof_value,
#                             const float *x1l, const float *y1l, const float *x1r, const float *y1r,
#                             const float *x2l, const float *y2l, const float *x2r, const float *y2r,
#                             const float time_resolution, const float dx, const float dy,
#                             const int nx, const int ny, const int event_num)

# TOF_dist_backprojection(float *image_bp, const float *proj_value, const float *tof_value,
#                             const float *x1l, const float *y1l, const float *x1r, const float *y1r,
#                             const float *x2l, const float *y2l, const float *x2r, const float *y2r,
#                             const float time_resolution, const float dx, const float dy,
#                             const int nx, const int ny, const int event_num)

# TOF_filter(float *filter_v, const int nx, const int ny, const float time_resolution)
