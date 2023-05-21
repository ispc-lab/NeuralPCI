# ==============================================================================
# Copyright (c) 2023 The NeuralPCI Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from data.MultiframeDataset import DHBDataset, NLDriveDataset
from model.NeuralPCI import NeuralPCI
from pytorch3d.loss import chamfer_distance
from utils.smooth_loss import compute_smooth
from utils.utils import *
sys.path.append('./utils/EMD')
sys.path.append('./utils/CD')
from emd import earth_mover_distance
import chamfer3D.dist_chamfer_3D


class Runner:
    def __init__(self, args):
        self.args = args
        # Init Dataset
        if args.dataset == 'DHB':
            self.dataset = DHBDataset(data_root = args.dataset_path, 
                                      scene_list = args.scenes_list, 
                                      interval = args.interval) 
        elif args.dataset == 'NL_Drive':
            self.dataset = NLDriveDataset(data_root = args.dataset_path, 
                                          scene_list = args.scenes_list, 
                                          interval = args.interval,
                                          num_points = args.num_points, 
                                          num_frames = args.num_frames)

        # Init DataLoader
        self.data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=4)

        self.chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()

        self.model = NeuralPCI(args).cuda()
        print("Number of parameters in model is {:.3f}M".format(sum(tensor.numel() for tensor in self.model.parameters())/1e6))

        # Loss recorder
        self.recorder = self.get_recorder(args)

        # Define time stamp for input and interpolation frames 
        self.time_seq, self.time_intp = self.get_timestamp(args)
        

    def get_recorder(self, args):
        recorder = {}
        recorder['loss_all_CD'] = []
        recorder['loss_all_EMD'] = []

        for i in range(args.interval - 1):
            recorder['loss_frame{}_CD'.format(i+1)] = []
            recorder['loss_frame{}_EMD'.format(i+1)] = []

        return recorder
        

    def get_timestamp(self, args):
        time_seq = [t for t in np.linspace(args.t_begin, args.t_end, args.num_frames)]
        t_left = time_seq[args.num_frames//2 - 1]
        t_right = time_seq[args.num_frames//2]
        time_intp = [t for t in np.linspace(t_left, t_right, args.interval+1)]
        time_intp = time_intp[1:-1]
        
        return time_seq, time_intp


    def loop(self):
        args = self.args
        print("Optimization Start for {} samples!".format(len(self.data_loader)))
        # get one sample from DataLoader (for-loop)
        for idx, (input, gt) in tqdm(enumerate(self.data_loader), total=len(self.data_loader), file=sys.stdout):
            print("\n[Sample: {}]".format(idx+1))
            # input point cloud frames (x,y,z)
            # input = [pc1, pc2, pc3, pc4], a list of numpy arrays, each point cloud shape is (N, 3)
            for i in range(len(input)):
                input[i] = input[i].squeeze(0).cuda().contiguous().float()
            for i in range(len(gt)):
                gt[i] = gt[i].squeeze(0).cuda().contiguous().float()

            # Definition
            model = NeuralPCI(args).cuda()
            
            best_model_weight = self.optimize_NeuralPCI(input, model, args)

            intp_cd_error, intp_emd_error = self.eval_NeuralPCI(input, gt, model, best_model_weight, args)

        # print overall average result
        self.print_result()


    def demo(self):
        args = self.args

        model = NeuralPCI(args).cuda()
        
        # input point cloud frames (x,y,z)
        # input = [pc1, pc2, pc3, pc4], a list of numpy arrays, each point cloud shape is (N, 3)
        input_path = args.input_path
        input = np.load(input_path).tolist()
        input = input.cuda().contiguous().float()

        self.time_seq, self.time_intp = self.get_timestamp(args)

        best_model_weight = self.optimize_NeuralPCI(input, model, args)

        pc_intp = self.infer_NeuralPCI(input, model, best_model_weight, args.time_pred)

        if args.save_path:
            if not os.path.exists(args.save_path):
                os.mkdirs(save_path)
            np.save(args.save_path + 'pc_intp.npy', pc_intp.detach().cpu().numpy())


    def get_optimizer(self, model, args):
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'sgdm':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
        elif args.optimizer == 'radam':
            optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'nadam':
            optimizer = torch.optim.NAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99, momentum=0.9, weight_decay=args.weight_decay)
        return optimizer


    def get_scheduler(self, optimizer, args):
        if args.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
        elif args.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=16, eta_min=1e-5)
        elif args.scheduler == 'cycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.iters)
        elif args.scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=100)
        return scheduler


    def compute_loss(self, pc_current, pc_pred, pc_input):  
        args = self.args
        loss = 0

        dist1, dist2, _, _ = self.chamLoss(pc_pred.unsqueeze(0), pc_input.unsqueeze(0))
        chamfer_loss = (dist1 + dist2).sum() * 0.5
        loss = loss + chamfer_loss * args.factor_cd

        if args.dataset == 'DHB':
            dist = earth_mover_distance(pc_pred.unsqueeze(0), pc_input.unsqueeze(0), transpose=False)
            emd_loss = (dist / pc_pred.shape[0]).mean()
            loss = loss + emd_loss * args.factor_emd

        elif args.dataset == 'NL_Drive':
            flow = pc_pred - pc_current
            smooth_loss = compute_smooth(pc_current.unsqueeze(0), flow.unsqueeze(0), k=9)
            smooth_loss = smooth_loss.squeeze(0).sum()
            loss = loss + smooth_loss * args.factor_smooth

        return loss


    def optimize_NeuralPCI(self, input, model, args):
        model.train()
        
        optimizer = self.get_optimizer(model, args)
        
        if args.scheduler:
            scheduler = self.get_scheduler(optimizer, args)

        # record optimization best result
        best_loss = np.inf
        best_iter = 0
        # optimize xx iterations for one sample
        for iter in range(args.iters):
            optimizer.zero_grad()
            loss = 0
            # input current point cloud to predict point cloud at time_pred
            for i in range(len(input)//2 - 1, len(input)//2 + 1):
                pc_current = input[i]
                time_current = self.time_seq[i]
                
                for j in range(len(input)):
                    time_pred = self.time_seq[j]
                    
                    pc_pred = pc_current + model(pc_current, time_current, time_pred)
                    
                    loss = loss + self.compute_loss(pc_current, pc_pred, input[j])

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_iter = iter
                best_model_weight = model.state_dict()
            
            if iter % 50 == 0:
                print(f"[Iter: {iter}] [Loss: {loss.item():.5f}]")

            if args.scheduler:
                if args.scheduler == 'plateau':
                    scheduler.step(loss.item())
                else:
                    scheduler.step()
        
        return best_model_weight


    def eval_NeuralPCI(self, input, gt, model, weight, args):
        model.eval()
        with torch.no_grad():
            model.load_state_dict(weight)
            
            pc_intp = []
            # Predict the inpolation frame at time_intp
            for j in range(len(self.time_intp)):
                pc_pred = []
                time_pred = self.time_intp[j]

                # Use nearest input frame as the reference
                for i in range(len(input)//2 - 1, len(input)//2 + 1):
                    pc_current = input[i]
                    time_current = self.time_seq[i]

                    pc_pred.append(pc_current + model(pc_current, time_current, time_pred))

                # NN-Intp
                if j < 0.5 * len(self.time_intp):
                    pc_intp.append(pc_pred[0].cuda())
                else:
                    pc_intp.append(pc_pred[1].cuda())
            
            intp_cd_error = []
            intp_emd_error = []
            for i in range(len(pc_intp)): 
                cd_error = chamfer_distance(pc_intp[i].unsqueeze(0), gt[i].unsqueeze(0))[0]

                dist = earth_mover_distance(pc_intp[i].unsqueeze(0), gt[i].unsqueeze(0), transpose=False)
                emd_error = (dist / pc_intp[i].shape[0]).mean()

                intp_cd_error.append(cd_error)
                intp_emd_error.append(emd_error)

            print("Eval: Average Interpolation CD Error: {}".format(torch.mean(torch.tensor(intp_cd_error))))
            print("Eval: Average Interpolation EMD Error: {}".format(torch.mean(torch.tensor(intp_emd_error))))
            
            self.recorder['loss_all_CD'].append(torch.mean(torch.tensor(intp_cd_error)).item())
            self.recorder['loss_all_EMD'].append(torch.mean(torch.tensor(intp_emd_error)).item())
            for i in range(args.interval - 1):
                self.recorder['loss_frame{}_CD'.format(i+1)].append(intp_cd_error[i].item())
                self.recorder['loss_frame{}_EMD'.format(i+1)].append(intp_emd_error[i].item())

            return intp_cd_error, intp_emd_error
        
    
    def print_result(self):
        recorder = self.recorder
        args = self.args
        print("\n=======================================")
        print("Final Result CD Loss is:{}".format(np.mean(recorder['loss_all_CD'])))
        print("Final Result EMD Loss is:{}".format(np.mean(recorder['loss_all_EMD'])))
        for i in range(args.interval - 1):
            print("=======================================")
            print("Final Frame-{} Result CD Loss is:{}".format(i+1, np.mean(recorder['loss_frame{}_CD'.format(i+1)])))
            print("Final Frame-{} Result EMD Loss is:{}".format(i+1, np.mean(recorder['loss_frame{}_EMD'.format(i+1)])))
        print("=======================================")


    def infer_NeuralPCI(self, input, model, weight, time_pred):
        with torch.no_grad():
            model.load_state_dict(weight)
            model.eval()
            
            if time_pred <= 0.5:
                pc_current = input[len(input)//2 - 1]
                time_current = i
            else:
                pc_current = input[len(input)//2]
                time_current = i
            pc_intp = pc_current + model(pc_current, time_current, time_pred)

        return pc_intp