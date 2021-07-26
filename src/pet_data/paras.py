import numpy as np 

class DirPara:
    def __init__(self, dir_paths, slice_list, time_resolution, counts):
        self.dir_path = dir_paths
        self.slice_list = slice_list
        self.time_resolution = time_resolution
        self.counts = counts



class Listmode: 
    def __init__(self, tof_value, 
                x1l, y1l, x1r, y1r,
                x2l, y2l, x2r, y2r, 
                time_resolution, event_num):
        self.tof_value = tof_value
        self.x1l = x1l
        self.y1l = y1l
        self.x1r = x1r
        self.y1r = y1r
        self.x2l = x2l
        self.y2l = y2l
        self.x2r = x2r
        self.y2r = y2r
        self.time_resolution = time_resolution
        self.event_num = event_num

class ImagePara:
    def __init__(self, dx, dy, nx, ny):
        self.dx = dx
        self.dy = dy
        self.nx = nx
        self.ny = ny

class NetworkInit:
    def __init__(self, s_factor, image_para, cuda_device,
                feature_channels, mid_channels, num_seq_blocks, num_iters):
        self.s_factor = s_factor
        self.image_para = image_para
        self.device =  cuda_device    
        self.feature_channels = feature_channels
        self.mid_channels = mid_channels
        self.num_seq_blocks =  num_seq_blocks
        self.num_iters = num_iters

class LearningRatePara:
    def __init__(self, init_lr, decay_weight, decay_interval, decay_mode):
        self.init_lr = init_lr
        self.decay_weight = decay_weight
        self.decay_interval = decay_interval
        self.decay_mode = decay_mode