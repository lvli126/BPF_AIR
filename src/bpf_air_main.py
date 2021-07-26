import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from networks.data_loader import MyTrainData
from networks.airnet_2d import AirNet2d
from trainers.trainer import train, re_train, test
from pet_data.paras import DirPara,ImagePara, NetworkInit, LearningRatePara
import numpy as np 
from tqdm import tqdm
import os

# gpu
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)    

# data dir
gt_path = "/home/lyuli/bpf-learning/PET_2nd_simu/xcat_2Dslice/"
simu_path = "/home/lyuli/bpf-learning/PET_2nd_simu/xcat_2Dsimu/"
dir_paths = [gt_path, simu_path]
slice_list = np.load("/home/lyuli/bpf-learning/PET_2nd_simu/test_index.npy").tolist()
time_resolution=200
counts = 1e6
train_dir_para = DirPara(dir_paths, slice_list, time_resolution, counts)
valid_dir_para = DirPara(dir_paths, slice_list, time_resolution, counts)
# image para
image_para = ImagePara(dx=3.125, dy=3.125, nx=200, ny=200)

# network initial parameters
network_init_para = NetworkInit(s_factor=0.1, image_para=image_para, cuda_device=device,
                                feature_channels=1, mid_channels=48, num_seq_blocks=3, 
                                num_iters=20)

initial_image = torch.ones([image_para.nx, image_para.ny], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

# learning rate para
lr_para = LearningRatePara(init_lr=0.01, decay_weight=0.8, decay_interval=500, decay_mode="simple_multi")

batch_size = 16
epoch_num = 500
save_model_interval = 10
result_path = "/home/lyuli/BPF_AIR/model_result/"
train(train_dir_para,valid_dir_para, network_init_para, initial_image, 
          lr_para, batch_size, epoch_num, save_model_interval, result_path,
          device)