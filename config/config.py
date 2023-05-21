import argparse

def npci_config():
    parser = argparse.ArgumentParser(description="Neural PCI")
    """Data"""
    parser.add_argument('--dataset', type=str, default='DHB', choices=['DHB', 'NL_Drive'], help='Dataset: DHB/NL_Drive.')
    parser.add_argument('--dataset_path', type=str, default='./dataset/DHB/', help='Dataset path.')
    parser.add_argument('--scenes_list', type=str, default='./data/scene_list.txt', help='Path of the scene list to be used in the dataset.')
    parser.add_argument('--interval', type=int, default=4, help='Interval frames between point cloud sequence.')
    parser.add_argument('--num_points', type=int, default=1024, help='Point number [default: 1024 for DHB / 8192 for NL_Drive].')
    parser.add_argument('--num_frames', type=int, default=4, help='Number of input point cloud frames.')
    parser.add_argument('--t_begin', type=float, default=0., help='Time stamp of the first input frame.')
    parser.add_argument('--t_end', type=float, default=1., help='Time stamp of the last input frame.')

    """Model"""
    parser.add_argument('--dim_pc', type=int, default=3, help='Dimension of point cloud spatial coordinate (xyz).')
    parser.add_argument('--dim_time', type=int, default=1, help='Dimension of point cloud temporal coordinate (t).')
    parser.add_argument('--layer_width', type=int, default=128, help='Layer width of NeuralPCI.')
    parser.add_argument('--norm', type=str, default=None, help='Normalization layer type.')
    parser.add_argument('--act_fn', type=str, default='leaky_relu', help='Activation function type.')
    parser.add_argument('--depth_encode', type=int, default=5, help='Depth of hidden layers for encoding spatio-temporal coordinate.')
    parser.add_argument('--depth_pred', type=int, default=1, help='Depth of hidden layers for predicting spatio-temporal coordinate.')
    parser.add_argument('--pe_mul', type=int, default=3, help='Dimension multiplied number of positional encoding.')
    parser.add_argument('--use_rrf', action='store_true', default=False, help='Whether to use random fourior feature transformation.')
    parser.add_argument('--dim_rrf', type=int, default=32, help='Dimension of random fourior feature.')
    parser.add_argument('--zero_init', action='store_true', default=False, help='Whether to initialize zero weight for the last layer.')

    """Optimization"""
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgdm', 'radam', 'nadam','rmsprop'], help='Optimizer type.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--scheduler', type=str, default=None, choices=['step', 'cosine', 'cycle', 'plateau'], help='Scheduler of learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0., help='Weight decay.')
    parser.add_argument('--iters', type=int, default=1000, metavar='N', help='Number of iterations to optimize the model.')
    parser.add_argument('--factor_cd', type=float, default=1., help='Chamfer Distance loss weight.')
    parser.add_argument('--factor_emd', type=float, default=50., help='Earth Movers Distance loss weight.')
    parser.add_argument('--factor_smooth', type=float, default=1., help='Smoothness loss weight.')
    
    """Demo"""
    parser.add_argument('--demo', action='store_true', default=False, help='Whether to use custom input point clouds for demo.')
    parser.add_argument('--input_path', type=str, default='./dataset/', help='Path to load input point cloud sequence (4 frames).')
    parser.add_argument('--intp_time', type=float, default=0.5, help='interpolation time (0-1) of output point cloud.')
    parser.add_argument('--save_path', type=str, default='./', help='Path to save output interpolation point cloud.')

    args = parser.parse_args()

    return args
