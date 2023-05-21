import torch
import numpy as np
import sys
sys.path.append('./utils/point_cloud_query')
sys.path.append('./utils/pointnet2')
from pointnet2_utils import grouping_operation
import points_query


def index_points_group(points, knn_idx):
    """
    Input:
        points: input points data, [B, N, C]
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    # import time
    # s = time.time()
    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = grouping_operation(points_flipped, knn_idx.int()).permute(0, 2, 3, 1)
    # e = time.time()
    # print('index_points_group', e-s)
    return new_points


def my_knn_query(k, query_pos, all_pos):
    """
    query_pos.shape = (b, sample, 3)
    all_pos.shape = (b, n, 3)
    return shape = (b, sample, k), (b, sample, k)
    """
    b, m, _ = query_pos.shape
    k_indices = torch.zeros((b, m, k), dtype=torch.int64, device='cuda:0')
    k_dis = torch.zeros((b, m, k), dtype=torch.float, device='cuda:0')
    
    points_query.knn_query(k, all_pos, query_pos, k_indices, k_dis)
    return k_indices, k_dis


def compute_smooth(pc1, pred_flow, k):
    """
    pc1: [B, N, 3]
    pred_flow: [B, N, 3]
    """
    kidx, _  = my_knn_query(k, pc1, pc1)
    grouped_flow = index_points_group(pred_flow, kidx)
    diff_flow = torch.norm(grouped_flow - pred_flow.unsqueeze(2), dim = 3).sum(dim = 2) / k
    return diff_flow