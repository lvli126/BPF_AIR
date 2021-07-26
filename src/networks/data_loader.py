import torch.utils.data as data
import torch
import os
import glob
import numpy as np 
from pet_data.paras import Listmode, ImagePara

'''
def get_dataset(dir_path, time_resolution, counts):
    listmode_path = dir_path[0]
    gt_path = dir_path[1]
    dataset = []
    listmode_sample_dir= listmode_path + "slice0"+f"/sub.{0}/lors_{time_resolution}ps.npy"
    listmode_sample = np.load(listmode_sample_dir)
    sub_num = counts // listmode_sample.shape[0] 

    for iterfile in glob.glob(os.path.join(gt_path, "*.bin")):
        slice_num = os.path.basename(iterfile)[5:-4]
        gt_name = "slice"+slice_num+".bin"
        gt_dir = os.path.join(gt_path, gt_name)
        
        listmode_dir_list = []
        for i in range(int(sub_num+1)):
            listmode_dir_list.append(listmode_path + "slice"+slice_num+f"/sub.{i}/lors_{time_resolution}ps.npy")
        dataset.append({"listmode": listmode_dir_list, "gt": gt_dir, "slice_num": int(slice_num)})
    return dataset
'''
def get_dataset(dir_path, slice_list, time_resolution, counts):
    listmode_path = dir_path[1]
    gt_path = dir_path[0]
    dataset = []
    listmode_sample_dir= listmode_path + "slice0"+f"/sub.{0}/lors_{time_resolution}ps.npy"
    listmode_sample = np.load(listmode_sample_dir)
    sub_num = counts // listmode_sample.shape[0] 

    for slice_num in slice_list:
        
        gt_name = f"slice{slice_num}.bin"
        gt_dir = os.path.join(gt_path, gt_name)
        
        listmode_dir_list = []
        for i in range(int(sub_num+1)):
            listmode_dir_list.append(listmode_path + f"slice{slice_num}/sub.{i}/lors_{time_resolution}ps.npy")
        dataset.append({"listmode": listmode_dir_list, "gt": gt_dir, "slice_num": int(slice_num)})
    return dataset

class MyTrainData(data.Dataset):
    def __init__(self, dir_para, gt_shape = (100,200), padding_for_gt = ((50,50),(0,0)), transform_flag=False):
        self.dir_path = dir_para.dir_path
        self.time_resolution = dir_para.time_resolution
        self.counts = int(dir_para.counts)
        self.dataset = get_dataset(dir_para.dir_path, dir_para.slice_list, dir_para.time_resolution, dir_para.counts)
        self.gt_shape = gt_shape
        self.padding_for_gt = padding_for_gt
        self.transform_flag = transform_flag
        

    def __getitem__(self, idx):
        listmode_dir = self.dataset[idx]["listmode"]
        gt_dir = self.dataset[idx]["gt"]
        per_listmode = int(self.counts / len(listmode_dir))
        slice_num = self.dataset[idx]["slice_num"]
        # per_sample_size = self.dataset[idx]["per_sample_size"]
        # load gt
        '''# ground_truth size为100x200
           # zero-padding to resize gt as 400x400
        '''
        ground_truth = np.fromfile(gt_dir,dtype=np.uint16).reshape(self.gt_shape).astype(np.float32) # ground_truth size为100x200
        ground_truth = np.pad(ground_truth, self.padding_for_gt, 'constant', constant_values=(0, 0))# zero-padding to resize gt as 400x400
        # load listmode
        listmode_data = np.empty([0, 9]) # listmode的shape是列
        for dir in listmode_dir:
            listmode = np.load(dir)[:per_listmode, :]
            listmode_data = np.vstack((listmode_data, listmode))

        # 构造Listmode类，作为listmode输入，需要提前将listmode_data转为torch.tensor
        listmode_data = torch.from_numpy(listmode_data).unsqueeze(0).unsqueeze(0)
    
        # 构造projection_data, 为每个lor的投影值
        projection_data = torch.ones(self.counts).unsqueeze(0).unsqueeze(0)

        ground_truth = ground_truth[np.newaxis,:,:].astype(np.float32)
        ground_truth = torch.from_numpy(ground_truth)

        

        return listmode_data, projection_data, ground_truth, slice_num, torch.tensor(self.time_resolution), torch.tensor(self.counts)

    def __len__(self) -> int:
        return len(self.dataset)

        
        