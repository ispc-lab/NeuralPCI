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

import torch
import torch.nn as nn
import numpy as np


def get_activation(activation):
    if activation.lower() == 'relu':
        return nn.ReLU(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif activation.lower() == 'sigmoid':
        return nn.Sigmoid()
    elif activation.lower() == 'softplus':
        return nn.Softplus()
    elif activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'mish':
        return nn.Mish(inplace=True)
    else:
        raise Exception("Activation Function Error")


def get_norm(norm, width):
    if norm == 'LN':
        return nn.LayerNorm(width)
    elif norm == 'BN':
        return nn.BatchNorm1d(width)
    elif norm == 'IN':
        return nn.InstanceNorm1d(width)
    elif norm == 'GN':
        return nn.GroupNorm(width)
    else:
        raise Exception("Normalization Layer Error")


class NeuralPCI_Layer(torch.nn.Module):
    def __init__(self, 
                 dim_in,
                 dim_out,
                 norm=None, 
                 act_fn=None
                 ):
        super().__init__()
        layer_list = []
        layer_list.append(nn.Linear(dim_in, dim_out))
        if norm:
            layer_list.append(get_norm(norm, dim_out))
        if act_fn:
            layer_list.append(get_activation(act_fn))
        self.layer = nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.layer(x)
        return x


class NeuralPCI_Block(torch.nn.Module):
    def __init__(self, 
                 depth, 
                 width,
                 norm=None, 
                 act_fn=None
                 ):
        super().__init__()
        layer_list = []
        for _ in range(depth):
            layer_list.append(nn.Linear(width, width))
            if norm:
                layer_list.append(get_norm(norm, width))
            if act_fn:
                layer_list.append(get_activation(act_fn))
        self.mlp = nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.mlp(x)
        return x


class NeuralPCI(torch.nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        dim_pc = args.dim_pc
        dim_time = args.dim_time
        layer_width = args.layer_width 
        act_fn = args.act_fn
        norm = args.norm
        depth_encode = args.depth_encode
        depth_pred = args.depth_pred
        pe_mul = args.pe_mul

        if args.use_rrf:
            dim_rrf = args.dim_rrf
            self.transform = 0.1 * torch.normal(0, 1, size=[dim_pc, dim_rrf]).cuda()
        else:
            dim_rrf = dim_pc

        # input layer
        self.layer_input = NeuralPCI_Layer(dim_in = (dim_rrf + dim_time) * pe_mul, 
                                           dim_out = layer_width, 
                                           norm = norm,
                                           act_fn = act_fn
                                           )

        # hidden layers
        self.hidden_encode = NeuralPCI_Block(depth = depth_encode, 
                                             width = layer_width, 
                                             norm = norm,
                                             act_fn = act_fn
                                             )

        # insert interpolation time
        self.layer_time = NeuralPCI_Layer(dim_in = layer_width + dim_time * pe_mul, 
                                          dim_out = layer_width, 
                                          norm = norm,
                                          act_fn = act_fn
                                          )

        # hidden layers
        self.hidden_pred = NeuralPCI_Block(depth = depth_pred, 
                                           width = layer_width, 
                                           norm = norm,
                                           act_fn = act_fn
                                           )

        # output layer
        self.layer_output = NeuralPCI_Layer(dim_in = layer_width, 
                                          dim_out = dim_pc, 
                                          norm = norm,
                                          act_fn = None
                                          )
        
        # zero init for last layer
        if args.zero_init:
            for m in self.layer_output.layer:
                if isinstance(m, nn.Linear):
                    # torch.nn.init.normal_(m.weight.data, 0, 0.01)
                    m.weight.data.zero_()
                    m.bias.data.zero_()
    
    def posenc(self, x):
        """
        sinusoidal positional encoding : N ——> 3 * N
        [x] ——> [x, sin(x), cos(x)]
        """
        sinx = torch.sin(x)
        cosx = torch.cos(x)
        x = torch.cat((x, sinx, cosx), dim=1)
        return x

    def forward(self, pc_current, time_current, time_pred):
        """
        pc_current: tensor, [N, 3]
        time_current: float, [1]
        time_pred: float, [1]
        output: tensor, [N, 3]
        """
        
        time_current = torch.tensor(time_current).repeat(pc_current.shape[0], 1).cuda().float().detach()
        time_pred = torch.tensor(time_pred).repeat(pc_current.shape[0], 1).cuda().float().detach()
        
        if self.args.use_rrf:
            pc_current = torch.matmul(2. * torch.pi * pc_current, self.transform)

        x = torch.cat((pc_current, time_current), dim=1)
        x = self.posenc(x)
        x = self.layer_input(x)

        x = self.hidden_encode(x)
        
        time_pred = self.posenc(time_pred)
        x = torch.cat((x, time_pred), dim=1)
        x = self.layer_time(x)

        x = self.hidden_pred(x)

        x = self.layer_output(x)
        
        return x