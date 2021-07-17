import torch
import numpy as np
import math
import time
from operators.pytorch.online2d.projector import Project, BackProject, BPF, ProjectFunction
from operators.pytorch.online2d import projector

def test_tof2d(torch_listmode, time_resolution, pixel_size, image_grid, device):
    
    tof_value = 2*torch_listmode[0,0,:, 8]
    x1l = torch_listmode[0,0,:, 0]
    y1l = torch_listmode[0,0,:, 1]
    x1r = torch_listmode[0,0,:, 2]
    y1r = torch_listmode[0,0,:, 3]
    x2l = torch_listmode[0,0,:, 4]
    y2l = torch_listmode[0,0,:, 5]
    x2r = torch_listmode[0,0,:, 6]
    y2r = torch_listmode[0,0,:, 7]
    dx,  dy = pixel_size[0], pixel_size[1]
    nx, ny = image_grid[0], image_grid[1]
    event_num = torch_listmode.size()[2]
    torch_projection = torch.ones(event_num).unsqueeze(0).unsqueeze(0).to(device)
    # torch_projection =  torch.tensor(torch_projection, device = device)
    # print(torch_listmode.size())
    # print(torch_projection.size())
    # print(x1l.size())

    bp = BackProject(tof_value, x1l, y1l, x1r, y1r,
                    x2l, y2l, x2r, y2r, time_resolution, dx, dy, nx, ny, event_num, device)
    proj = Project(tof_value, x1l, y1l, x1r, y1r,
                    x2l, y2l, x2r, y2r, time_resolution, dx, dy, nx, ny, event_num, device)
    # image_bp = bp(torch_projection)
    # # print(image_bp.size())
    # # print(image_bp.detach().cpu().numpy()[0,0,90:100,90:100])
    # np.save("/home/lyuli/gitpackages/test_data/bp.npy", np.squeeze(image_bp.detach().cpu().numpy()))

    # bpf = BPF(tof_value, x1l, y1l, x1r, y1r,
    #                 x2l, y2l, x2r, y2r, time_resolution, dx, dy, nx, ny, event_num, device)
    # image_bpf  = bpf(torch_projection)
    # # print(image_bpf.size())
    # np.save("/home/lyuli/gitpackages/test_data/bpf.npy", np.squeeze(image_bpf.detach().cpu().numpy()))
    iter_num = 10
    bp_v = np.zeros((iter_num, 200, 200))
    proj_v =  np.zeros((iter_num, event_num))
    
    for iter in range(iter_num):
        torch_bp = bp(torch_projection)
        torch_projection =  proj(torch_bp)
        bp_v[iter,:,:]=np.squeeze(torch_bp.detach().cpu().numpy())
        proj_v[iter,:]=np.squeeze(torch_projection.detach().cpu().numpy())

    np.save("/home/lyuli/gitpackages/test_data/bp_v.npy", bp_v)
    np.save("/home/lyuli/gitpackages/test_data/proj_v.npy", proj_v)



if __name__ == '__main__':
    device = torch.device("cuda:1")
    torch.cuda.set_device(1)
    data_dir  =  "/home/lyuli/bpf-learning/PET_2nd_simu/xcat_2Dsimu/slice30/sub.6/lors_200ps.npy"
    listmode  = np.load(data_dir)[:50000,:]
    torch_listmode = torch.tensor(listmode).unsqueeze(0).unsqueeze(0)
    # print(torch_listmode.size())
    time_resolution = torch.tensor(200)
    pixel_size  = torch.tensor([3.125, 3.125])
    image_grid = torch.tensor([200,200])
    test_tof2d(torch_listmode, time_resolution, pixel_size, image_grid, device)
    
