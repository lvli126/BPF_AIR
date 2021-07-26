
import torch
import numpy as np
import math
import time

# from torch._C import float32       
from operators.pytorch.online2d.projector1 import Project, BackProject, BPF
from pet_data.paras import Listmode, ImagePara
# from operators.pytorch.online2d import projector

def test_tof2d(torch_listmode, time_resolution, pixel_size, image_grid, device):
    
    event_num = torch.tensor(torch_listmode.size()[2])
    dx,  dy = pixel_size[0], pixel_size[1]
    nx, ny = image_grid[0], image_grid[1]

    torch_listmode = torch_listmode
    tof_value = 2*torch_listmode[:,:,:, 8]
    x1l = torch_listmode[:,:,:, 0]
    y1l = torch_listmode[:,:,:, 1]
    x1r = torch_listmode[:,:,:, 2]
    y1r = torch_listmode[:,:,:, 3]
    x2l = torch_listmode[:,:,:, 4]
    y2l = torch_listmode[:,:,:, 5]
    x2r = torch_listmode[:,:,:, 6]
    y2r = torch_listmode[:,:,:, 7]
    
    torch_projection = torch.ones(event_num).unsqueeze(0).unsqueeze(0).to(device)
    s_factor=0.3
    # torch_projection =  torch.tensor(torch_projection, device = device)
    # print(torch_listmode.size())
    # print(torch_projection.size())
    # print(x1l.size())

    bp = BackProject(tof_value, x1l, y1l, x1r, y1r,
                    x2l, y2l, x2r, y2r, time_resolution, dx, dy, nx, ny, event_num, device)
    # proj = Project(tof_value, x1l, y1l, x1r, y1r,
    #                 x2l, y2l, x2r, y2r, time_resolution, dx, dy, nx, ny, event_num, device)
    # bpf = BPF(tof_value, x1l, y1l, x1r, y1r,
    #                 x2l, y2l, x2r, y2r, time_resolution, dx, dy, nx, ny, event_num, device)
    # bpmapping = BpMapping(s_factor, image_para, device)

    # # test bpmapping
    
    # image1 = torch.ones((200,200)).unsqueeze(0).unsqueeze(0).to(device) # 用全1作为初始值
    # image2 = bpmapping(image1, torch_projection, listmode)
    # print(listmode.event_num)
    # np.save("/home/lyuli/gitpackages/test_data/image2_pytorch2.npy", np.squeeze(image2.detach().cpu().numpy()))

    # test bp
    # bp = BackProject(image_para, device)
    image_bp = bp(torch_projection)
    # # print(image_bp.size())
    # # print(image_bp.detach().cpu().numpy()[0,0,90:100,90:100])
    np.save("/home/lyuli/gitpackages/test_data/bp_pytorch3.npy", np.squeeze(image_bp.detach().cpu().numpy()))

    # test bpf
    # image_bpf  = bpf(torch_projection)
    # # print(image_bpf.size())
    # np.save("/home/lyuli/gitpackages/test_data/bpf.npy", np.squeeze(image_bpf.detach().cpu().numpy()))
    

    # test proj
    # proj_v =  np.zeros(event_num)
    # one_map = torch.ones((200,200)).unsqueeze(0).unsqueeze(0).to(device)
    # torch_projection =  proj(one_map)
    # proj_v=np.squeeze(torch_projection.detach().cpu().numpy())
    # np.save("/home/lyuli/gitpackages/test_data/proj_pytorch.npy", proj_v)

    # test mlem
    # iter_num=3
    # bp_v = torch.ones((iter_num,200,200)).unsqueeze(0).unsqueeze(0).to(device)
    # # proj_v =  torch.tensor(np.zeros((iter_num, event_num)))
    # measured_proj = torch.ones((event_num,1), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    # emap = np.fromfile("/home/lyuli/bpf-learning/PET_2nd_simu/emap.bin",dtype=np.float32).reshape(200,200)
    # torch_emap = torch.tensor(emap).unsqueeze(0).unsqueeze(0).to(device)
    # proj_ratio = torch.zeros_like(measured_proj)
    # print(measured_proj.size())
    # for iter in range(iter_num):
    #     torch_projection =  proj(bp_v[:,:,iter,:,:])
    #     # print(torch_projection[0,0,:,0].size())
    #     proj_ratio =  torch.div(measured_proj.float(), torch_projection) #measured_proj[0,0,:,0]/ torch_projection[0,0,:,0] #
    #     bp_ratio = bp(proj_ratio)
    #     bp_v[:,:,iter+1, :,:] = bp_v[:,:,iter,:,:] / torch_emap * bp_ratio
        
    # #     # bp_v[iter,:,:]=np.squeeze(torch_bp.detach().cpu().numpy())
    # #     # proj_v[iter,:]=np.squeeze(torch_projection.detach().cpu().numpy())

    # np.save("/home/lyuli/gitpackages/test_data/mlem_pytorch.npy", np.squeeze(bp_v.detach().cpu().numpy()))
    # # np.save("/home/lyuli/gitpackages/test_data/proj_pytorch.npy", proj_v)

    

if __name__ == '__main__':
    device = torch.device("cuda:1")
    torch.cuda.set_device(1)
    # print(torch.__version__)
    data_dir  =  "/home/lyuli/bpf-learning/PET_2nd_simu/xcat_2Dsimu/slice30/sub.6/lors_200ps.npy"
    listmode  = np.load(data_dir)[:50000,:]
    print(listmode[:10,0])
    torch_listmode = torch.tensor(listmode,dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    # print(torch_listmode.size())
    time_resolution = torch.tensor(200)
    pixel_size  = torch.tensor([3.125, 3.125])
    image_grid = torch.tensor([200,200])
    test_tof2d(torch_listmode, time_resolution, pixel_size, image_grid, device)
    
