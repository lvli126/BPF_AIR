import numpy as np 
import math 
import ctypes
from ctypes import *
from funcs import get_TOF_dist_backprojection, get_TOF_dist_projection, get_TOF_filter

__get_TOF_dist_backprojection = get_TOF_dist_backprojection()
__get_TOF_dist_projection = get_TOF_dist_projection()
# __get_TOF_filter = get_TOF_filter()




def TOF_dist_projection(image: np.ndarray, listmode: np.ndarray, time_resolution: np.float32, pixel_size: np.ndarray):

    event_num = np.uint32(listmode.shape[0])
    proj_v = np.zeros(event_num, dtype= np.float32)

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

def TOF_dist_backprojection(proj_v: np.ndarray, listmode: np.ndarray, time_resolution: np.float32, image_grid: np.ndarray, pixel_size: np.ndarray):
    # 生成变量
    event_num = listmode.shape[0]
    # proj_v = np.ones((event_num,1), dtype= np.float32)
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
    
# def TOF_filter(nx: np.uint32, ny: np.uint32, tof_sigma: np.float32):
#     total_size = int(nx*ny)
#     filter_v = np.zeros((total_size,1),dtype=np.float32)

#     filter_value = filter_v.ctypes.data_as(POINTER(c_float))
#     __get_TOF_filter(filter_value, nx, ny, tof_sigma)

#     filter_v = filter_v.reshape((int(nx),int(ny)))
#     return filter_v
def TOF_filter(nx: np.uint32, ny: np.uint32,  tof_sigma: np.float32):

    import scipy.special
    
    nx2 = nx / 2
    ny2 = ny / 2
    x_ = y_ = np.linspace(-nx2 + 0.5, nx2 - 0.5, nx)/ nx2# 与ramp一样，将采样点用物理尺寸标准化
    xx, yy = np.meshgrid(x_, y_, indexing = 'ij')
    w0 = xx ** 2 + yy ** 2 # 频域w的平方
    tmp = w0*(np.pi * tof_sigma)**2 
    filter_v = scipy.special.ive(0,tmp)**(-1)#直接用iv函数容易出现math range error，如下一行的ive代替，ive为指数修正的0阶第一类贝塞尔函数


    return filter_v

def TOF_BPF(listmode: np.ndarray, time_resolution: np.float32, image_grid: np.ndarray, pixel_size: np.ndarray):
    nx, ny = image_grid[0], image_grid[1]
    dx, dy = pixel_size[0], pixel_size[1]
    img_bp = TOF_dist_backprojection(listmode, time_resolution, image_grid, pixel_size)
    img_bp_norm = img_bp / dx / dy
    tof_sigma = time_resolution * 0.3 / 2 / 2.355 / dx
    tof_filter = TOF_filter(nx, ny, tof_sigma)
    image_recon = np.real(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(img_bp_norm))*tof_filter*nx)))
    return image_recon



if __name__ == '__main__':
    import time
    file_path = "/home/lyuli/bpf-learning/PET_2nd_simu/xcat_2Dsimu/slice30/sub.6/lors_200ps.npy"
    # file_path = "/home/lvli/Documents/gitpackages/test/lors_200ps.npy"
    listmode = np.load(file_path)[:50000,:]
    emap = np.fromfile("/home/lyuli/bpf-learning/PET_2nd_simu/emap.bin",dtype=np.float32).reshape(200,200)
    time_resolution = 200
    image_grid = np.array([200,200])
    pixel_size = np.array([3.125,3.125])
    start = time.time()
    tof_sigma = time_resolution * 0.3 / 2 / 2.355 / pixel_size[0]
    measured_proj = np.ones(listmode.shape[0])
    # filters = TOF_filter(image_grid[0],image_grid[1], tof_sigma)
    
    # test bp
    # image_bp = TOF_dist_backprojection(measured_proj, listmode, time_resolution, image_grid, pixel_size)
    
    # test_proj
    one_map = np.ones((200,200))
    proj_value =  TOF_dist_projection(one_map, listmode, time_resolution, pixel_size)
    np.save("/home/lyuli/gitpackages/test_data/proj_cuda.npy", proj_value)
    #     
    # test mlem
    # iter_num = 4
    # image_bp = np.ones((iter_num+1,200,200))
    # proj_value = np.zeros((iter_num,listmode.shape[0]))
    
    # for i in range(iter_num):
    #     proj_value[i,:] =  TOF_dist_projection(image_bp[i,:,:], listmode, time_resolution, pixel_size)[:,0]
    #     proj_ratio = np.divide(measured_proj,proj_value[i,:], out=np.zeros(listmode.shape[0]), where=proj_value[i,:]!=0)
    #     bp_ratio = TOF_dist_backprojection(proj_ratio, listmode, time_resolution, image_grid, pixel_size)
    #     image_bp[i+1,:,:] = image_bp[i,:,:] / emap * bp_ratio
    # # image_recon = TOF_BPF(listmode, time_resolution, image_grid, pixel_size)
    # end = time.time()
    # print(end-start)
    # np.save("/home/lyuli/gitpackages/test_data/bp_cuda.npy", image_bp)
    # np.save("/home/lyuli/gitpackages/test_data/proj_cuda.npy", proj_value)

    # np.save("/home/lvli/Documents/gitpackages/test/filters.npy", filters)
    # np.save("/home/lvli/Documents/gitpackages/test/image_bp.npy", image_bp)
    # np.save("/home/lvli/Documents/gitpackages/test/image_recon.npy", image_recon)