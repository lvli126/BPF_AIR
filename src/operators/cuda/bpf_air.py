import numpy as np 
import math 
import ctypes
from ctypes import *
from funcs import get_TOF_dist_backprojection, get_TOF_dist_projection, get_TOF_filter

__get_TOF_dist_backprojection = get_TOF_dist_backprojection()
__get_TOF_dist_projection = get_TOF_dist_projection()
__get_TOF_filter = get_TOF_filter()

def TOF_dist_backprojection(listmode: np.ndarray, time_resolution: np.float32, image_grid: np.ndarray, pixel_size: np.ndarray):
    # 生成变量
    event_num = listmode.shape[0]
    proj_v = np.ones((event_num,1), dtype= np.float32)
    total_size = image_grid[0] * image_grid[1]
    image = np.zeros((total_size,1),dtype= np.float32)

    # 将变量转为c程序需要的格式
    x_bp = image.ctypes.data_as(POINTER(c_float))
    proj_value = proj_v.ctypes.data_as(POINTER(c_float))
    tof_value = (np.array(2*listmode[:,8])).ctypes.data_as(POINTER(c_float))
    x1l = listmode[:,0].ctypes.data_as(POINTER(c_float))
    y1l = listmode[:,1].ctypes.data_as(POINTER(c_float))
    x1r = listmode[:,2].ctypes.data_as(POINTER(c_float))
    y1r = listmode[:,3].ctypes.data_as(POINTER(c_float))
    x2l = listmode[:,4].ctypes.data_as(POINTER(c_float))
    y2l = listmode[:,5].ctypes.data_as(POINTER(c_float))
    x2r = listmode[:,6].ctypes.data_as(POINTER(c_float))
    y2r = listmode[:,7].ctypes.data_as(POINTER(c_float))
    time_resolution = float(time_resolution)
    dx, dy = float(pixel_size[0]), float(pixel_size[1])
    nx,ny = np.uint32(image_grid[0]), np.uint32(image_grid[1])
    event_num = np.uint32(event_num)
    
    
    # 运行函数
    __get_TOF_dist_backprojection(x_bp, proj_value, tof_value, 
                                    x1l,y1l, x1r, y1r, x2l,y2l,x2r,y2r, 
                                    time_resolution, dx, dy, nx, ny, event_num)
    
    image = image.reshape((int(nx), int(ny)))
    # x_bp = x_bp.reshape(int(nx),int(ny))
    return image


def TOF_dist_projection(image: np.ndarray, listmode: np.ndarray, time_resolution: np.float32, pixel_size: np.ndarray):

    event_num = np.uint32(listmode.shape[0])
    proj_v = np.zeros((event_num,1), dtype= np.float32)

    x_p = image.ctypes.data_as(POINTER(c_float))
    proj_value = proj_v.ctypes.data_as(POINTER(c_float))
    tof_value = (np.array(2*listmode[:,8])).ctypes.data_as(POINTER(c_float))
    x1l = listmode[:,0].ctypes.data_as(POINTER(c_float))
    y1l = listmode[:,1].ctypes.data_as(POINTER(c_float))
    x1r = listmode[:,2].ctypes.data_as(POINTER(c_float))
    y1r = listmode[:,3].ctypes.data_as(POINTER(c_float))
    x2l = listmode[:,4].ctypes.data_as(POINTER(c_float))
    y2l = listmode[:,5].ctypes.data_as(POINTER(c_float))
    x2r = listmode[:,6].ctypes.data_as(POINTER(c_float))
    y2r = listmode[:,7].ctypes.data_as(POINTER(c_float))
    time_resolution = float(time_resolution)
    dx, dy = float(pixel_size[0]), float(pixel_size[1])
    nx,ny = np.uint32(image.shape[0]), np.uint32(image.shape[1])

    __get_TOF_dist_projection(proj_value, x_p, tof_value, 
                              x1l,y1l,x1r, y1r, x2l,y2l,x2r,y2r,
                              time_resolution, dx, dy, nx, ny, event_num)

    return proj_v
    
def TOF_filter(nx: np.uint32, ny: np.uint32, time_resolution: np.float32):
    total_size = int(nx*ny)
    filter_v = np.zeros((total_size,1),dtype=np.float32)

    filter_value = filter_v.ctypes.data_as(POINTER(c_float))
    __get_TOF_filter(filter_value, nx, ny, time_resolution)

    filter_v.reshape((nx,ny))
    return filter_v


if __name__ == '__main__':
    # file_path = "/home/lyuli/bpf-learning/PET_2nd_simu/xcat_2Dsimu/slice30/sub.6/lors_200ps.npy"
    file_path = "/home/lvli/Documents/gitpackages/test/lors_200ps.npy"
    listmode = np.load(file_path)[:1,:]
    time_resolution = 200
    image_grid = np.array([200,200])
    pixel_size = np.array([3.125,3.125])
    image_bp = TOF_dist_backprojection(listmode, time_resolution, image_grid, pixel_size)
    np.save("/home/lvli/Documents/gitpackages/test/image_bp.npy", image_bp)