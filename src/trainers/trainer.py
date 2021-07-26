import numpy as np 
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from networks.data_loader import MyTrainData
import os
from networks.airnet_2d import AirNet2d
import torch.nn as nn
from torch import nn, optim
from pet_data.paras import Listmode
# metrics
# 1: psnr
# 2: ssim 调用torch自带包

def psnr(y_predb, yb):
    from psnr import psnr
    
    psnr_sum = 0.0
    for i_slice in range(yb.shape[0]):
        psnr_sum += psnr(yb[i_slice:i_slice+1, :, :, :],y_predb[i_slice:i_slice+1, :, :, :])
    return psnr_sum

def learning_rate_update_by_step(step, decay_weight, decay_interval, optimizer):
    if step % decay_interval == int(decay_interval-1):
        for group in optimizer.param_groups:
            group['lr'] *= decay_weight
 
def learning_rate_update_by_epoch(epoch, decay_weight, decay_interval, optimizer):
    for group in optimizer.param_groups:
        group['lr'] *= decay_weight**(epoch // decay_interval)
 

def learning_rate_update(step, decay_weight, decay_interval, optimizer, mode="simple_multi"):
    if mode=="simple_multi":
        learning_rate_update_by_step(step, decay_weight, decay_interval, optimizer)
    if mode=="exp_multi":
        learning_rate_update_by_epoch(step, decay_weight, decay_interval, optimizer)

def validation(model, loss_fn, valid_dataloaders, batch_size, initial_image, device):
    valid_loss = 0
    psnr_value = 0
    model.eval()
    with torch.no_grad():
        for j, data in enumerate(valid_dataloaders,0):
            listmode_data, projection_data, ground_truth, slice_num, time_resolution, counts = data

            # 将tensor移动到配置好的设备上（GPU）
            listmode_data = listmode_data.to(device)
            labels = ground_truth.to(device)
            time_resolution = time_resolution.to(device)
            counts = counts.to(device)
            
            # 将listmode封装成Listmode类，便于传输
            listmodes = Listmode(listmode_data[:,:,:,8],
                                listmode_data[:,:,:,0],listmode_data[:,:,:,1],listmode_data[:,:,:,2],listmode_data[:,:,:,3],
                                listmode_data[:,:,:,4],listmode_data[:,:,:,5],listmode_data[:,:,:,6],listmode_data[:,:,:,7],
                                time_resolution, counts)

            # 预测
            outputs = model(initial_image, projection_data, listmodes)
            loss = loss_fn(outputs, labels)

            # calculate metrics
            valid_loss += loss.item()
            psnr_value += psnr(outputs, labels).item()
        
        avg_valid_loss = valid_loss/(j+1)/batch_size
        avg_psnr = psnr_value/(j+1)/batch_size
    return avg_valid_loss, avg_psnr

def training(model, loss_fn, optimizer, train_dataloaders, batch_size, initial_image,lr_para, device):
    train_loss=0
    model.train()
    for i, data in enumerate(train_dataloaders,0):
        listmode_data, projection_data, ground_truth, slice_num, time_resolution, counts = data

        # 将tensor移动到配置好的设备上（GPU）
        listmode_data = listmode_data.to(device)
        labels = ground_truth.to(device)
        time_resolution = time_resolution.to(device)
        counts = counts.to(device)
        
        # 将listmode封装成Listmode类，便于传输
        listmodes = Listmode(listmode_data[:,:,:,8],
                            listmode_data[:,:,:,0],listmode_data[:,:,:,1],listmode_data[:,:,:,2],listmode_data[:,:,:,3],
                            listmode_data[:,:,:,4],listmode_data[:,:,:,5],listmode_data[:,:,:,6],listmode_data[:,:,:,7],
                            time_resolution, counts)

        # 前向传播
        outputs = model(initial_image, projection_data, listmodes)
        loss = loss_fn(outputs, labels)

        # 反向传播和optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 学习率更新
        learning_rate_update(i, lr_para.decay_weight, lr_para.decay_interval, optimizer, mode=lr_para.decay_mode)
        train_loss += loss.item()    
    avg_train_loss = train_loss / (i+1)/batch_size
    return avg_train_loss,optimizer

def train_loop(epoch_start, epoch_num, model, loss_fn, optimizer, 
               train_dataloaders, valid_dataloaders, batch_size, initial_image,
               lr_para, save_model_interval, result_path, device):
    
    for epoch in tqdm(range(epoch_start, int(epoch_start + epoch_num))):
        
        #####################
        #训练
        #####################
        avg_train_loss,optimizer = training(model, loss_fn, optimizer, train_dataloaders, batch_size, initial_image, lr_para, device)

        #####################
        #验证
        #####################
        avg_valid_loss, avg_psnr = validation(model, loss_fn, valid_dataloaders, batch_size, initial_image, device)

        #####################
        #画图
        #####################
        writer.add_scalars('MSE loss',{"train": avg_train_loss, "valid": avg_valid_loss},epoch)
        
        # writer.add_scalar('ssim',ssim_value,epoch)
        writer.add_scalar('psnr',avg_psnr,epoch)      
        
        if ((epoch+1) % save_model_interval)==0:
            model_dir = result_path+f"/model_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict() },
                 model_dir)

        print(f"epoch: {(epoch+1)}, "
              f'loss: {avg_valid_loss:.3f}, '
            #   f'ssim: {ssim_value:.5f}, '
              f'psnr: {avg_psnr:.5f}, ')


def train(train_dir_para,valid_dir_para, network_init_para, initial_image, 
          lr_para, batch_size, epoch_num, save_model_interval, result_path,
          device):
    train_dataset = MyTrainData(train_dir_para, gt_shape = (100,200), padding_for_gt = ((50,50),(0,0)), transform_flag=True)
    valid_dataset = MyTrainData(valid_dir_para, gt_shape = (100,200), padding_for_gt = ((50,50),(0,0)), transform_flag=True) 

    train_dataloaders = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloaders = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = AirNet2d(network_init_para)
    loss_fn = nn.MSELoss(reduction='mean')
    
    optimizer = optim.Adam(model.parameters(),lr=lr_para.init_lr)
    epoch_start = 0
    train_loop(epoch_start, epoch_num, model, loss_fn, optimizer, 
            train_dataloaders, valid_dataloaders, batch_size, initial_image,
            lr_para, save_model_interval, result_path, device)

    
def re_train(train_dir_para,valid_dir_para, network_init_para, initial_image, 
          lr_para, batch_size, epoch_num, save_model_interval, result_path,
          model_path,device):
    train_dataset = MyTrainData(train_dir_para, gt_shape = (100,200), padding_for_gt = ((50,50),(0,0)), transform_flag=True)
    valid_dataset = MyTrainData(valid_dir_para, gt_shape = (100,200), padding_for_gt = ((50,50),(0,0)), transform_flag=True) 

    train_dataloaders = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloaders = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = AirNet2d(network_init_para)
    load_model = torch.load(model_path)  
    model.load_state_dict(load_model['model_state_dict'])
    epoch_start = load_model['epoch']
    lr=load_model['optimizer_state_dict']['param_groups'][0]['lr']
    optimizer = optim.Adam(model.parameters(),lr=lr)
    optimizer.load_state_dict(load_model['optimizer_state_dict'])
    loss_fn = nn.MSELoss(reduction='mean')
    train_loop(epoch_start, epoch_num, model, loss_fn, optimizer, 
            train_dataloaders, valid_dataloaders, batch_size, initial_image,
            lr_para, save_model_interval, result_path, device)

def test(test_dir_para,network_init_para, initial_image, model_path, epoch_index, result_path, device):
    test_dataset = MyTrainData(test_dir_para, gt_shape = (100,200), padding_for_gt = ((50,50),(0,0)), transform_flag=True)
    test_dataloaders = DataLoader(test_dataset, batch_size=1, shuffle=False)

    test_loss = 0
    psnr_value = 0
    output_path = result_path+f"epoch{epoch_index}/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model = AirNet2d(network_init_para)
    load_model = torch.load(model_path)  
    model.load_state_dict(load_model['model_state_dict'])
    model.eval()
    loss_fn = nn.MSELoss(reduction="mean")
    with torch.no_grad():
        for i, data in enumerate(test_dataloaders,0):      
            listmode_data, projection_data, ground_truth, slice_num, time_resolution, counts = data

            # 将tensor移动到配置好的设备上（GPU）
            listmode_data = listmode_data.to(device)
            labels = ground_truth.to(device)
            time_resolution = time_resolution.to(device)
            counts = counts.to(device)
            
            # 将listmode封装成Listmode类，便于传输
            listmodes = Listmode(listmode_data[:,:,:,8],
                                listmode_data[:,:,:,0],listmode_data[:,:,:,1],listmode_data[:,:,:,2],listmode_data[:,:,:,3],
                                listmode_data[:,:,:,4],listmode_data[:,:,:,5],listmode_data[:,:,:,6],listmode_data[:,:,:,7],
                                time_resolution, counts)

            # 预测
            outputs = model(initial_image, projection_data, listmodes)
            loss = loss_fn(outputs, labels)

            # calculate metrics
            test_loss += loss.item()
            psnr_value += psnr(outputs, labels).item()
            
            np.squeeze(outputs.detach().cpu().numpy()).tofile(output_path+f"output{slice_num.numpy()}.bin")
        test_loss = test_loss/(i+1)
        # ssim_value = ssim_value/(i+1)
        psnr_value = psnr_value/(i+1)
        print(f'test loss: {test_loss:.5f}, test psnr: {psnr_value:.5f}')


    



        


