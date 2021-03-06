import torch
from torch.autograd import grad
import torch.nn as nn
import numpy as np
import math
import time
import torch.fft as fft
import online2d


class ProjectFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, tof_value, 
                x1l, y1l, x1r, y1r,
                x2l, y2l, x2r, y2r,
                time_resolution, dx, dy, nx, ny, event_num):
        ctx.tof_value = tof_value
        ctx.x1l = x1l
        ctx.y1l = y1l
        ctx.x1r = x1r
        ctx.y1r = y1r
        ctx.x2l = x1l
        ctx.y2l = y1l
        ctx.x2r = x1r
        ctx.y2r = y1r
        ctx.time_resolution = time_resolution
        ctx.dx = dx
        ctx.dy = dy
        ctx.nx = nx
        ctx.ny = ny
        ctx.event_num = event_num
        projection_data = online2d.project(image, tof_value, 
                                            x1l, y1l, x1r, y1r,
                                            x2l, y2l, x2r, y2r,
                                            time_resolution, dx, dy, nx, ny, event_num)
        # ctx.save_for_backward()
        return projection_data

    @staticmethod
    def backward(ctx, grad_projection):
        grad_image = online2d.backproject(grad_projection.contiguous(), ctx.tof_value, 
                                          ctx.x1l, ctx.y1l, ctx.x1r, ctx.y1r,
                                          ctx.x2l, ctx.y2l, ctx.x2r, ctx.y2r,
                                          ctx.time_resolution, ctx.dx, ctx.dy, ctx.nx, ctx.ny, ctx.event_num)
        
        return grad_image

class BackProjectFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projection_data, tof_value,
                x1l, y1l, x1r, y1r,
                x2l, y2l, x2r, y2r,
                time_resolution, dx, dy, nx, ny, event_num):
        ctx.tof_value = tof_value
        ctx.x1l = x1l
        ctx.y1l = y1l
        ctx.x1r = x1r
        ctx.y1r = y1r
        ctx.x2l = x1l
        ctx.y2l = y1l
        ctx.x2r = x1r
        ctx.y2r = y1r
        ctx.time_resolution = time_resolution
        ctx.dx = dx
        ctx.dy = dy
        ctx.nx = nx
        ctx.ny = ny
        ctx.event_num = event_num
        back_image = online2d.backproject(projection_data, tof_value, 
                                            x1l, y1l, x1r, y1r,
                                            x2l, y2l, x2r, y2r,
                                            time_resolution, dx, dy, nx, ny, event_num)
        # ctx.save_for_backward(back_image)
        return back_image
    
    @staticmethod
    def backward(ctx, grad_image):
        grad_projection = online2d.project(grad_image.contiguous(), ctx.tof_value, 
                                          ctx.x1l, ctx.y1l, ctx.x1r, ctx.y1r,
                                          ctx.x2l, ctx.y2l, ctx.x2r, ctx.y2r,
                                          ctx.time_resolution, ctx.dx, ctx.dy, ctx.nx, ctx.ny, ctx.event_num)
        return grad_projection




class Project(torch.nn.Module):
    def  __init__(self, tof_value, 
                x1l, y1l, x1r, y1r,
                x2l, y2l, x2r, y2r,
                time_resolution, dx, dy, nx, ny, event_num, cuda_device):
        super(Project,  self).__init__()
        
        self.tof_value = torch.tensor(tof_value, device=cuda_device)
        self.x1l = torch.tensor(x1l, device=cuda_device)
        self.y1l = torch.tensor(y1l, device=cuda_device)
        self.x1r = torch.tensor(x1r, device=cuda_device)
        self.y1r = torch.tensor(y1r, device=cuda_device)
        self.x2l = torch.tensor(x2l, device=cuda_device)
        self.y2l = torch.tensor(y2l, device=cuda_device)
        self.x2r = torch.tensor(x2r, device=cuda_device)
        self.y2r = torch.tensor(y2r, device=cuda_device)
        self.time_resolution =  torch.tensor(time_resolution,device=cuda_device)
        self.dx = torch.tensor(dx, device=cuda_device)
        self.dy = torch.tensor(dy, device=cuda_device)
        self.nx = torch.tensor(nx, device=cuda_device)
        self.ny = torch.tensor(ny, device=cuda_device)
        self.event_num =  torch.tensor(event_num, device=cuda_device)
        
    def forward(self, image):
        return  ProjectFunction.apply(image, self.tof_value, 
                                 self.x1l, self.y1l, self.x1r, self.y1r,
                                 self.x2l, self.y2l, self.x2r, self.y2r,
                                 self.time_resolution,  self.dx, self.dy, 
                                 self.nx, self.ny, self.event_num)

class BackProject(torch.nn.Module):
    def __init__(self, tof_value, 
                x1l, y1l, x1r, y1r,
                x2l, y2l, x2r, y2r,
                time_resolution, dx, dy, nx, ny, event_num, cuda_device):
        super(BackProject,  self).__init__()
        # self.projection_data = projection_data
        self.tof_value = torch.tensor(tof_value, device=cuda_device)
        self.x1l = torch.tensor(x1l, device=cuda_device)
        self.y1l = torch.tensor(y1l, device=cuda_device)
        self.x1r = torch.tensor(x1r, device=cuda_device)
        self.y1r = torch.tensor(y1r, device=cuda_device)
        self.x2l = torch.tensor(x2l, device=cuda_device)
        self.y2l = torch.tensor(y2l, device=cuda_device)
        self.x2r = torch.tensor(x2r, device=cuda_device)
        self.y2r = torch.tensor(y2r, device=cuda_device)
        self.time_resolution =  torch.tensor(time_resolution,device=cuda_device)
        self.dx = torch.tensor(dx, device=cuda_device)
        self.dy = torch.tensor(dy, device=cuda_device)
        self.nx = torch.tensor(nx, device=cuda_device)
        self.ny = torch.tensor(ny, device=cuda_device)
        self.event_num =  torch.tensor(event_num)

    def forward(self, projection_data):
        return BackProjectFunction.apply(projection_data, self.tof_value, 
                                 self.x1l, self.y1l, self.x1r, self.y1r,
                                 self.x2l, self.y2l, self.x2r, self.y2r,
                                 self.time_resolution,  self.dx, self.dy, 
                                 self.nx, self.ny, self.event_num)



class BPF(torch.nn.Module):
    
    def  __init__(self, tof_value, 
                x1l, y1l, x1r, y1r,
                x2l, y2l, x2r, y2r,
                time_resolution, dx, dy, nx, ny, event_num, cuda_device):
        super(BPF, self).__init__()
        # self.projection_data = projection_data
        self.tof_value = torch.tensor(tof_value, device=cuda_device)
        self.x1l = torch.tensor(x1l, device=cuda_device)
        self.y1l = torch.tensor(y1l, device=cuda_device)
        self.x1r = torch.tensor(x1r, device=cuda_device)
        self.y1r = torch.tensor(y1r, device=cuda_device)
        self.x2l = torch.tensor(x2l, device=cuda_device)
        self.y2l = torch.tensor(y2l, device=cuda_device)
        self.x2r = torch.tensor(x2r, device=cuda_device)
        self.y2r = torch.tensor(y2r, device=cuda_device)
        self.time_resolution =  torch.tensor(time_resolution,device=cuda_device)
        self.dx = torch.tensor(dx, device=cuda_device)
        self.dy = torch.tensor(dy, device=cuda_device)
        self.nx = torch.tensor(nx, device=cuda_device)
        self.ny = torch.tensor(ny, device=cuda_device)
        self.event_num =  torch.tensor(event_num, device=cuda_device)
        self.device = cuda_device
        # print(cuda_device)

    
    def forward(self, projection_data):
        back_image =  BackProjectFunction.apply(projection_data, self.tof_value, 
                                 self.x1l, self.y1l, self.x1r, self.y1r,
                                 self.x2l, self.y2l, self.x2r, self.y2r,
                                 self.time_resolution,  self.dx, self.dy, 
                                 self.nx, self.ny, self.event_num)
        norm_back_image =  back_image / self.dx / self.dy
        freq_back_image =  fft.fftshift(fft.fft2(norm_back_image, dim=(-2,-1), norm=None), dim=(-2,-1))
        
        # define filter
        tof_sigma = self.time_resolution * 0.3 / 2 / 2.355 / self.dx
        nx2 =  self.nx / 2
        # print(nx2.device)
        x_ = y_ = torch.linspace(-nx2 + 0.5, nx2 - 0.5, self.nx , device = self.device) / nx2# ???ramp?????????????????????????????????????????????
        xx, yy = torch.meshgrid(x_, y_)
        w0 = torch.pow(xx, 2) + torch.pow(yy, 2) # ??????w?????????
        tmp = w0 * torch.pow((torch.tensor(math.pi) * tof_sigma), 2) 
        freq_filter = torch.special.i0e(tmp)**(-1)#?????????iv??????????????????math range error??????????????????ive?????????ive??????????????????0???????????????????????????
        
        # filtering and ifft
        image = fft.ifft2(fft.ifftshift(freq_filter * freq_back_image * self.nx, dim=(-2,-1)), dim=(-2, -1)).real
        return image

##################################################################
# the combination modules: x^(n+1/2) = x^(n) - s*BPF((Ax^(n) - y))
#####################################################################

class BpMappingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s_factor, image, projection_data, tof_value, 
                x1l, y1l, x1r, y1r, x2l, y2l, x2r, y2r,
                time_resolution, dx, dy, nx, ny, event_num):
        # Ax-y
        # projection_value = torch.ones((event_num,1))
        projection_new = online2d.project(image, tof_value, 
                                            x1l, y1l, x1r, y1r,
                                            x2l, y2l, x2r, y2r,
                                            time_resolution, dx, dy, nx, ny, event_num)

        projection_diff = projection_new - projection_data

        # A-transpose
        back_diff = online2d.backproject(projection_diff, tof_value, 
                                            x1l, y1l, x1r, y1r,
                                            x2l, y2l, x2r, y2r,
                                            time_resolution, dx, dy, nx, ny, event_num)
        s_diff = torch.mul(back_diff, s_factor)
        result = image - s_diff
        ctx.save_for_backward(s_factor, back_diff, tof_value, 
                              x1l, y1l, x1r, y1r, x2l, y2l, x2r, y2r,
                              time_resolution, dx, dy, nx, ny, event_num)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        [s_factor, back_diff, tof_value, 
        x1l, y1l, x1r, y1r, x2l, y2l, x2r, y2r,
        time_resolution, dx, dy, nx, ny, event_num] = ctx.saved_tensors

        grad_proj = online2d.project(grad_output.contiguous(), tof_value, 
                                    x1l, y1l, x1r, y1r,
                                    x2l, y2l, x2r, y2r,
                                    time_resolution, dx, dy, nx, ny, event_num)

        grad_image = online2d.backproject(grad_proj.contiguous(), tof_value, 
                                          x1l, y1l, x1r, y1r,
                                          x2l, y2l, x2r, y2r,
                                          time_resolution, dx, dy, nx, ny, event_num)
        
        grad_input = grad_output - torch.mul(grad_image.contiguous(), s_factor)
        grad_s = - torch.sum(torch.sum(torch.sum(torch.sum(torch.mul(back_diff, grad_output), -1),-1),-1),-1) / (grad_image.shape[0])
        return grad_input, grad_s

class BPMapping(torch.nn.Module):
    def __int__(self,tof_value, 
                x1l, y1l, x1r, y1r,
                x2l, y2l, x2r, y2r,
                time_resolution, dx, dy, nx, ny, event_num, cuda_device, s_factor=0.0):
        super(BPMapping, self).__init__()
        self.tof_value = torch.tensor(tof_value).to(cuda_device)
        self.x1l = torch.tensor(x1l, device=cuda_device)
        self.y1l = torch.tensor(y1l, device=cuda_device)
        self.x1r = torch.tensor(x1r, device=cuda_device)
        self.y1r = torch.tensor(y1r, device=cuda_device)
        self.x2l = torch.tensor(x2l, device=cuda_device)
        self.y2l = torch.tensor(y2l, device=cuda_device)
        self.x2r = torch.tensor(x2r, device=cuda_device)
        self.y2r = torch.tensor(y2r, device=cuda_device)
        self.time_resolution =  torch.tensor(time_resolution,device=cuda_device)
        self.dx = torch.tensor(dx, device=cuda_device)
        self.dy = torch.tensor(dy, device=cuda_device)
        self.nx = torch.tensor(nx, device=cuda_device)
        self.ny = torch.tensor(ny, device=cuda_device)
        self.event_num =  torch.tensor(event_num, device=cuda_device)
        self.device = cuda_device
        self.s_factor = torch.nn.Parameter(torch.tensor([s_factor],dtype=torch.float).to(cuda_device))

    def forward(self, image, projection_data):
        return BpMappingFunction.apply(self.s_factor, image, projection_data, self.tof_value, 
                                 self.x1l, self.y1l, self.x1r, self.y1r,
                                 self.x2l, self.y2l, self.x2r, self.y2r,
                                 self.time_resolution,  self.dx, self.dy, 
                                 self.nx, self.ny, self.event_num)
