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


if __name__ == '__main__':
    # logging.basicConfig(filename='test2.log', format='%(message)s', level=logging.INFO)
    
    b, m, n = 8, 600, 2400
    k = 16
    all_pos = torch.randn((b, n, 3), device='cuda:0')
    query_index = torch.randint(0, 2400, (600, ), device='cuda:0')
    query_pos = all_pos[:, query_index, :]
    
    k_indices_1, k_dis_1 = knn_query(k, query_pos, all_pos)
    # logging.info(k_indices)
    # logging.info(k_dis)
    # logging.info('------------------------------------')
    
    k_indices_2, k_dis_2 = my_knn_query(k, query_pos, all_pos)
    # logging.info(k_indices)
    # logging.info(k_dis)
    
    # 看两个knn得到结果是否相同
    mask_1 = k_indices_1 - k_indices_2
    mask_2 = k_dis_1 - k_dis_2
    
    if mask_1.sum() == 0:
        print('yes_1')
    print(mask_2.sum())
    