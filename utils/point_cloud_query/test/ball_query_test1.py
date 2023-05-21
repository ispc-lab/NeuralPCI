import torch
import points_query
import logging


@torch.no_grad()
def get_square_distance(query_pos, all_pos):
    """
    query_pos.shape = (b, sample, 3)
    all_pos.shape = (b, n, 3)
    return shape = (b, sample, n)
    """
    res = ((query_pos.unsqueeze(dim=2) - all_pos.unsqueeze(dim=1)) ** 2).sum(dim=-1)
    return res


def ball_query(radius, k, query_pos, all_pos):
    """
    query_pos.shape = (b, sample, 3)
    all_pos.shape = (b, n, 3)
    all_x.shape = (b, n, c)
    return shape = (b, sample, k, 3), (b, sample, k, c), (b, sample, k)
    """
    square_dis = get_square_distance(query_pos, all_pos)
    k_dis, k_indices = square_dis.topk(k, largest=False)   # k_indices.shape = (b, sample, k)
    
    # ball query比knn麻烦一点
    mask = (k_dis > radius**2)
    temp = k_indices[:, :, 0:1].expand(-1, -1, k)
    temp2 = k_dis[:, :, 0:1].expand(-1, -1, k)
    
    k_indices[mask] = temp[mask]
    k_dis[mask] = temp2[mask]
    
    return k_indices, k_dis


def my_ball_query(radius, k, query_pos, all_pos):
    """
    query_pos.shape = (b, sample, 3)
    all_pos.shape = (b, n, 3)
    return shape = (b, sample, k), (b, sample, k)
    """
    b, m, _ = query_pos.shape
    k_indices = torch.zeros((b, m, k), dtype=torch.int64, device='cuda:0')
    k_dis = torch.zeros((b, m, k), dtype=torch.float, device='cuda:0')
    
    points_query.ball_query(k, radius, all_pos, query_pos, k_indices, k_dis)
    return k_indices, k_dis


if __name__ == '__main__':
    # logging.basicConfig(filename='test1.log', format='%(message)s', level=logging.INFO)
    
    b, m, n = 8, 600, 2400
    k = 16
    radius = 0.5
    query_pos = torch.randn((b, m, 3), device='cuda:0')
    all_pos = torch.randn((b, n, 3), device='cuda:0')
    
    k_indices_1, k_dis_1 = ball_query(radius, k, query_pos, all_pos)
    # logging.info(k_indices)
    # logging.info(k_dis)
    # logging.info('------------------------------------')
    
    k_indices_2, k_dis_2 = my_ball_query(radius, k, query_pos, all_pos)
    # logging.info(k_indices)
    # logging.info(k_dis)
    
    # 看两个knn得到结果是否相同
    mask_1 = k_indices_1 - k_indices_2
    mask_2 = k_dis_1 - k_dis_2
    
    if mask_1.sum() == 0:
        print('yes_1')
    print(mask_2.sum())
    