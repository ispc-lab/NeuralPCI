import torch
import numpy as np
import sys
sys.path.append('./utils/point_cloud_query')
import points_query


def fuse_interpolation(args, pc_pred):
    pred1 = pc_pred[0]
    pred2 = pc_pred[1]
    #fusion
    pred_fuse = torch.cat((pred1, pred2), dim=0)
    interp_idx = np.random.choice(args.num_points*2, args.num_points, replace=False)
    pc_interp = pred_fuse[interp_idx, :]

    return pc_interp

# fuse 4 point cloud prediction to output the interpolation result
# def fuse_interpolation(args, pc_pred):
#     pred1 = pc_pred[0]
#     pred2 = pc_pred[1]
#     pred3 = pc_pred[2]
#     pred4 = pc_pred[3]
#     #fusion
#     pred_fuse = torch.cat((pred1, pred2, pred3, pred4), dim=0)
#     interp_idx = np.random.choice(args.num_points*4, args.num_points, replace=False)
#     pc_interp = pred_fuse[interp_idx, :]

#     return pc_interp

def voxel_fuse_interpolation(point_cloud_list, num_points, leaf_size = 0.15):
    point_clouds = torch.cat(point_cloud_list, axis =0)
    point_cloud = point_clouds.cpu().numpy()

    filtered_points = []
    # calculate edge value
    x_min, y_min, z_min = np.amin(point_cloud, axis=0) #x,y,z max and min value
    x_max, y_max, z_max = np.amax(point_cloud, axis=0)
 
    # voxel grid dimension
    Dx = (x_max - x_min)//leaf_size + 1
    Dy = (y_max - y_min)//leaf_size + 1
    Dz = (z_max - z_min)//leaf_size + 1
    print("Dx x Dy x Dz is {} x {} x {}".format(Dx, Dy, Dz))
 
    # voxel index for each point
    h = list()  # h list of index
    for i in range(len(point_cloud)):
        hx = (point_cloud[i][0] - x_min) // leaf_size
        hy = (point_cloud[i][1] - y_min) // leaf_size
        hz = (point_cloud[i][2] - z_min) // leaf_size
        h.append(hx + hy * Dx + hz * Dx * Dy)
    h = np.array(h)
 
    # sort out point
    h_indice = np.argsort(h) # sort index of h
    h_sorted = h[h_indice]
    begin = 0
    for i in range(len(h_sorted)-1):   # 0~9999
        if h_sorted[i] == h_sorted[i + 1]:
            continue
        else:
            point_idx = h_indice[begin : i + 1]
            filtered_points.append(np.mean(point_cloud[point_idx], axis=0))
            begin = i + 1
 
    # to numpy
    filtered_points = np.array(filtered_points, dtype=np.float64)
    # random sample to wanted points number
    num = filtered_points.shape[0]
    print(num)
    # filtered_points_idx = []
    if num >= num_points:
        filtered_points_idx = np.random.choice(num, num_points, replace=False)
    else:
        filtered_points_idx = np.concatenate((np.arange(num), np.random.choice(num, num_points - num, replace=True)), axis = -1)
    filtered_points = filtered_points[filtered_points_idx, :]
    # np.save('fuse.npy', filtered_points)
    return torch.from_numpy(filtered_points).float()
    # return filtered_points



@torch.no_grad()
def get_square_distance(query_pos, all_pos):
    """
    query_pos.shape = (b, sample, 3)
    all_pos.shape = (b, n, 3)
    return shape = (b, sample, n)
    """
    res = ((query_pos.unsqueeze(dim=2) - all_pos.unsqueeze(dim=1)) ** 2).sum(dim=-1)
    return res


def knn_query(k, query_pos, all_pos):
    """
    query_pos.shape = (b, sample, 3)
    all_pos.shape = (b, n, 3)
    return shape = (b, sample, k), (b, sample, k)
    """
    square_dis = get_square_distance(query_pos, all_pos)
    k_dis, k_indices = square_dis.topk(k, largest=False)   # k_indices.shape = (b, sample, k)
    
    return k_indices, k_dis


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

# def auto_label(gt, intp, label_intp, k=1):
#     """
#     gt: numpy array, [1,N,3]        # in-between(gt) frame without label
#     intp: numpy array, [1,N,3]      # interpolation frame with label inherented from key frame
#     label_intp: numpy array, [N]    # label of key frame(consistent with interpolation frame)
    
#     return: numpy array, [N]        # label of in-between(gt) frame
#     """
#     # auto-labelling with kNN (nearest or voting)
#     k_indices, k_dis = knn_query(k, torch.from_numpy(gt).unsqueeze(0), torch.from_numpy(intp).unsqueeze(0))
#     # print(k_indices)
#     # print(k_indices.shape)
#     label_gt = label_intp[k_indices.squeeze(0).squeeze(1).numpy()]
#     # print(label_gt)
#     label_gt_list = []
#     for i in range(label_gt.shape[0]):
#         if k != 1:
#             label_gt_list.append(np.argmax(np.bincount(np.asarray(label_gt[i,:], dtype=int))))
#         else:
#             label_gt_list.append(label_gt[i])
    
#     return np.asarray(label_gt_list)

def auto_label(gt, intp, label_intp, k=1):
    """
    gt: tensor, [1,N,3]          # in-between(gt) frame without label
    intp: tensor, [1,N,3]        # interpolation frame with label inherented from key frame
    label_intp: tensor, [N]      # label of key frame(consistent with interpolation frame)
    
    return: numpy array, [N]     # label of in-between(gt) frame
    """
    # auto-labelling with kNN (nearest or voting)
    # k_indices, k_dis = knn_query(k, gt.unsqueeze(0), intp.unsqueeze(0))    # pytorch version
    k_indices, k_dis = my_knn_query(k, gt.unsqueeze(0), intp.unsqueeze(0))   # cuda version
    # print(k_indices)
    # print(k_indices.shape)
    label_gt = label_intp[k_indices.squeeze(0).squeeze(1)]
    # print(label_gt)
    label_gt_list = []
    for i in range(label_gt.shape[0]):
        if k != 1:
            label_gt_list.append(np.argmax(np.bincount(label_gt[i,:].cpu().numpy())))
        else:
            label_gt_list.append(label_gt[i].cpu().numpy())
    
    return np.asarray(label_gt_list)