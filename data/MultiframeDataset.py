import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class NLDriveDataset(Dataset):
    """
    Args:
        data_root: path for NL-Drive dataset
        scene_list: path of point cloud sequence list to load samples
        interval: point cloud sequence downsampling interval, pick one frame from every (interval) frames,
                  i.e. (interval - 1) interpolation frame between every two frames [default: 4]
        num_points: sample a fixed number of points in each input and gt point cloud frame [default: 8192]
        num_frames: number of input point cloud frames [default: 4]
    """
    def __init__(self, data_root, scene_list, interval=4, num_points=8192, num_frames=4):
        super(NLDriveDataset, self).__init__()
        self.data_root = data_root
        self.num_points = num_points
        self.scene_list = scene_list
        self.interval = interval
        self.num_frames = num_frames
        self.velodynes = self.read_scene_list()
    
    def read_scene_list(self):
        velodynes = []
        with open(self.scene_list, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n').split(' ')
                velodynes.append(line)
        return velodynes

    def __getitem__(self, index):
        sample_names = self.velodynes[index]
        pc = []
        gt = []
        pc_points_idx = []
        gt_points_idx = []

        for i in range(self.num_frames):
            # load data
            pc_path = os.path.join(self.data_root, sample_names[i])
            # print(pc_path)
            pc_raw = np.fromfile(pc_path, dtype = np.float32, count = -1).reshape([-1, 3])
            pc.append(pc_raw)

            # sample n_points
            num = pc_raw.shape[0]
            if num >= self.num_points:
                pc_points_idx.append(np.random.choice(num, self.num_points, replace=False))
            else:
                pc_points_idx.append(np.concatenate((np.arange(num), np.random.choice(num, self.num_points - num, replace=True)), axis = -1))
        
        num_gt = len(sample_names) - self.num_frames
        gt_intv = num_gt // (self.interval - 1)
        for i in range(self.interval-1):
            # load data and rm_ground if needed
            gt_path = os.path.join(self.data_root, sample_names[3 + (i+1)*gt_intv])
            # print(gt_path)
            gt_raw = np.fromfile(gt_path, dtype = np.float32, count = -1).reshape([-1, 3])
            gt.append(gt_raw)

            # sample n_points
            num = gt_raw.shape[0]
            if num >= self.num_points:
                gt_points_idx.append(np.random.choice(num, self.num_points, replace=False))
            else:
                gt_points_idx.append(np.concatenate((np.arange(num), np.random.choice(num, self.num_points - num, replace=True)), axis = -1))

        pc_sampled = []
        gt_sampled = []

        for i in range(self.num_frames):
            pc_sampled.append(pc[i][pc_points_idx[i], :].astype('float32'))
        for i in range(self.interval-1):
            gt_sampled.append(gt[i][gt_points_idx[i], :].astype('float32'))

        input = []
        gt = []
        for pc in pc_sampled:
            input.append(torch.from_numpy(pc))
        for intp in gt_sampled:
            gt.append(torch.from_numpy(intp))
        return input, gt

    def __len__(self):
        return len(self.velodynes)



class DHBDataset(Dataset):
    """
    data_root: path for DHB dataset
    scene_list: path of point cloud sequence list to load samples
    interval: point cloud sequence downsampling interval, pick one frame from every (interval) frames,
              i.e. (interval - 1) interpolation frame between every two frames  [default: 4]
    """
    def __init__(self, data_root, scene_list, interval=4): 
        self.data_root = data_root
        self.interval = interval
        self.scene_8IVFB = ['longdress','loot', 'redandblack', 'soldier']
        self.scenes = self.read_scene_list(scene_list)
        self.total = 0
        self.dataset_dict, self.dataset_scene_len = self.make_dataset()
        
    def read_scene_list(self, scenes_list):
        # read .txt file containing train/val scene number
        scenes = []
        with open(scenes_list, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                scenes.append(line)
        return scenes
    
    def make_dataset(self):
        scene_8IVFB = ['longdress','loot', 'redandblack', 'soldier']
        dataset_dict = {}
        dataset_scene_len = {}
        scene_ini = 0
        scene_end = 0
        for scene in self.scenes:
            if scene in self.scene_8IVFB:
                dataset_dict[scene] = self.get_rich_data_8ivfb(scene)
            else:
                dataset_dict[scene] = self.get_rich_data(scene)
            sample_len = dataset_dict[scene][-1]
            scene_end += sample_len
            dataset_scene_len[scene]=[scene_ini,scene_end]
            scene_ini = scene_end
        self.total = scene_end
        return dataset_dict, dataset_scene_len # [scene_num, 3], i.e. each scene has [data_tensor,GroupIdx,sample_len]

    def get_rich_data(self, scene):
        data_tensor = torch.load( os.path.join(self.data_root,scene+'_fps1024_aligned.pt') )[:, :, :]
        GroupIdx,sample_len=self.get_one_scene_index(len(data_tensor))
        # print('scene ====',scene, 'len seq ====', len(data_tensor))
        return [data_tensor,GroupIdx,sample_len]

    def get_rich_data_8ivfb(self, scene):
        _path = './'
        data_tensor = torch.load( os.path.join(self.data_root,scene+'.pt') )[:, :, :]
        GroupIdx,sample_len=self.get_one_scene_index(len(data_tensor))
        # print('scene ====',scene, 'len seq ====', len(data_tensor))
        return [data_tensor, GroupIdx, sample_len]

    def get_one_scene_index(self, len_):
        GroupIdx={}
        GroupIdx['pc1']=[]
        GroupIdx['pc2']=[]
        GroupIdx['pc3']=[]
        GroupIdx['pc4']=[]
        for k in range(self.interval - 1):
            GroupIdx[f'gt{k}']=[]
        ini_idx = 0
        end_idx = 0
        while ini_idx + self.interval * 3 < len_:
            end_idx = ini_idx + self.interval * 3
            GroupIdx['pc1'].append(ini_idx)
            GroupIdx['pc2'].append(ini_idx + self.interval)
            GroupIdx['pc3'].append(ini_idx + self.interval * 2)
            GroupIdx['pc4'].append(ini_idx + self.interval * 3)
            for k in range(self.interval - 1):
                GroupIdx[f'gt{k}'].append(ini_idx + self.interval + k + 1)
            ini_idx += self.interval
        assert(len(GroupIdx['pc1'])==len(GroupIdx['pc2']))
        assert(len(GroupIdx['pc1'])==len(GroupIdx['pc3']))
        assert(len(GroupIdx['pc1'])==len(GroupIdx['pc4']))
        assert(len(GroupIdx['pc1'])==len(GroupIdx['gt0']))
        sample_len = len(GroupIdx['pc1'])
        return GroupIdx, sample_len


    def pc_normalize(self, pc, max_for_the_seq):
        pc = pc.numpy()
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = max_for_the_seq
        pc = pc / m
        return pc

    def __getitem__(self, index):
        sample={}
        sample['indices']=[]

        for scene, ini_end in self.dataset_scene_len.items():
            if index < ini_end[1]:
                [data_tensor,GroupIdx,sample_len] = self.dataset_dict[scene]
                sample['scene']= scene
                inside_idx = index-ini_end[0]
                for pos, scene_sample_idx in GroupIdx.items():        
                    sample_idx = scene_sample_idx[inside_idx] # list of sample_idx
                    sample['indices'].append(sample_idx) #filled indices
                    pc = data_tensor[sample_idx]
                    if sample['scene'] in self.scene_8IVFB:
                        pc = self.pc_normalize(pc, max_for_the_seq=583.1497484423953)
                        pc = torch.from_numpy(pc)
                    sample[pos] = pc #filled pc
                sample['indices'] = np.array(sample['indices'])
                
                pc1 = sample["pc1"] # [1024, 3]
                pc2 = sample["pc2"]
                pc3 = sample["pc3"]
                pc4 = sample["pc4"]
                input = [pc1,pc2,pc3,pc4]
                gt = []
                for i in range(self.interval-1):
                    gt.append(sample[f'gt{i}'])

                return input, gt
                
    def __len__(self):
        return self.total


