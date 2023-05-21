import os
import numpy as np
import glob
import argparse
from tqdm import tqdm
from nuscenes import NuScenes
from pyquaternion import Quaternion
import av2.utils.io as io_utils

def parse_arg():
    parser = argparse.ArgumentParser(description="Preprocess Parameter")
    parser.add_argument('--root', type=str, default='', help='Dataset path.')
    parser.add_argument('--save_path', type=str, default='', help='Path to save processed dataset')
    parser.add_argument('--scenes_list', type=str, default='', help='Path of the scene list to be used in the dataset.')
    parser.add_argument('--dataset', type=str, default='', help='which dataset')
    parser.add_argument('--min_dist', type=float, default=3.0, help='Points < Min dist will be removed.')
    parser.add_argument('--max_dist', type=float, default=50.0, help='Points > Max dist will be removed.')
    parser.add_argument('--max_height', type=float, default=4.0, help='Points > Max height will be removed.')
    parser.add_argument('--remove', action='store_true', help='whether rm')
    args = parser.parse_args()

    return args


def my_ransac(data,
              distance_threshold=0.3,
              P=0.99,
              sample_size=3,
              max_iterations=10000,
              ):
    """
    :param data:
    :param sample_size:
    :param P :
    :param distance_threshold:
    :param max_iterations:
    :return:
    """
    # np.random.seed(12345)
    random.seed(12345)
    max_point_num = -999
    i = 0
    K = 10
    L_data = len(data)
    R_L = range(L_data)

    while i < K:
        s3 = random.sample(R_L, sample_size)

        if abs(data[s3[0],1] - data[s3[1],1]) < 3:
            continue
        
        coeffs = estimate_plane(data[s3,:], normalize=False)
        if coeffs is None:
            continue

        r = np.sqrt(coeffs[0]**2 + coeffs[1]**2 + coeffs[2]**2 )
        d = np.divide(np.abs(np.matmul(coeffs[:3], data.T) + coeffs[3]) , r)
        d_filt = np.array(d < distance_threshold)
        near_point_num = np.sum(d_filt,axis=0)

        if near_point_num > max_point_num:
            max_point_num = near_point_num

            best_model = coeffs
            best_filt = d_filt

            w = near_point_num / L_data

            wn = np.power(w, 3)
            p_no_outliers = 1.0 - wn
            
            K = (np.log(1-P) / np.log(p_no_outliers))

        i += 1
        if i > max_iterations:
            print(' RANSAC reached the maximum number of trials.')
            break

    return np.argwhere(best_filt).flatten(), best_model


def estimate_plane(xyz, normalize=True):
    """
    :param xyz:  3*3 array
    x1 y1 z1
    x2 y2 z2
    x3 y3 z3
    :return: a b c d
      model_coefficients.resize (4);
      model_coefficients[0] = p1p0[1] * p2p0[2] - p1p0[2] * p2p0[1];
      model_coefficients[1] = p1p0[2] * p2p0[0] - p1p0[0] * p2p0[2];
      model_coefficients[2] = p1p0[0] * p2p0[1] - p1p0[1] * p2p0[0];
      model_coefficients[3] = 0;
      // Normalize
      model_coefficients.normalize ();
      // ... + d = 0
      model_coefficients[3] = -1 * (model_coefficients.template head<4>().dot (p0.matrix ()));
    """
    vector1 = xyz[1,:] - xyz[0,:]
    vector2 = xyz[2,:] - xyz[0,:]

    if not np.all(vector1):
        print('will divide by zero..')
        return None
    dy1dy2 = vector2 / vector1

    if  not ((dy1dy2[0] != dy1dy2[1])  or  (dy1dy2[2] != dy1dy2[1])):
        return None

    a = (vector1[1]*vector2[2]) - (vector1[2]*vector2[1])
    b = (vector1[2]*vector2[0]) - (vector1[0]*vector2[2])
    c = (vector1[0]*vector2[1]) - (vector1[1]*vector2[0])
    # normalize
    if normalize:
        r = np.sqrt(a ** 2 + b ** 2 + c ** 2)
        a = a / r
        b = b / r
        c = c / r
    d = -(a*xyz[0,0] + b*xyz[0,1] + c*xyz[0,2])
    # return a,b,c,d
    return np.array([a,b,c,d])


class PreprocessNuscenes:
    def __init__(self, root, save_path, scenes_list, min_dist = 3.0, max_dist = 50.0, max_height = 4.0):
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.max_height = max_height
        self.root = root
        self.save_path = save_path
        self.scenes = self.read_scene_list(scenes_list) # scene-0001 ...
        self.max_points = 0
        self.min_points = 10**10
        self.save_processed()
        
        
    def read_scene_list(self, scenes_list):
        # read .txt file containing train/val scene number
        scenes = []
        with open(scenes_list, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                scenes.append(line)
        return scenes

    def load_calibrated_para(self, cur_scene, filename):
        for scene in self.nusc.scene:
            if scene['name'] == cur_scene:
                start_sample_rec = self.nusc.get('sample', scene['first_sample_token'])
                sd_rec = self.nusc.get('sample_data', start_sample_rec['data']['LIDAR_TOP'])

                # Make list of frames
                cur_sd_rec = sd_rec
                sd_tokens = []
                sd_tokens.append(cur_sd_rec['token'])
                while cur_sd_rec['next'] != '':
                    cur_sd_rec = self.nusc.get('sample_data', cur_sd_rec['next'])
                    sd_tokens.append(cur_sd_rec['token'])
                    if cur_sd_rec['filename'] == filename:
                        break
                break

        sc_rec = self.nusc.get('sample_data', sd_tokens[-1])
        lidar_token = sc_rec['token']
        lidar_rec = self.nusc.get('sample_data', lidar_token)
        calibrated = self.nusc.get('calibrated_sensor', lidar_rec['calibrated_sensor_token'])
        rot = np.array(calibrated['rotation'])
        trl = np.array(calibrated['translation'])
        return rot, trl
    
    def load_scene_frames(self, scene):
        # load frames of the scene and return file list [num_scene_frames]
        file = os.path.join('./scene-split', scene + '.txt')
        timestamps = []
        filenames = []
        with open(file) as f:
            for line in f.readlines():
                line = line.strip('\n').split(' ')
                filename = line[0]
                timestamp = float(line[1])
                filenames.append(filename)
                timestamps.append(timestamp)
                rot, trl = self.load_calibrated_para(scene, filename)
        return timestamps, filenames, rot, trl # rot and trl is consitent within a scene

    def save_processed(self):
        file_path = os.path.join(self.root, 'sweeps', 'LIDAR_TOP')
        idx = 1

        for scene in tqdm(self.scenes):
            print(scene)
            if idx < 81:
                print("skip", idx)
                idx += 1
                continue
            print("begin", idx)
            self.nusc = NuScenes(version = 'v1.0-trainval', dataroot=self.root, verbose=True)
            timestamps, filenames, rot, trl = self.load_scene_frames(scene)
            fn_times = []
            scene_save_path = os.path.join(self.save_path, 'scene-split', scene + '.txt')
            for i in tqdm(range(len(filenames))):
                # load .pcd data
                pc_path = file_path + '/' + filenames[i]
                data_save_path = os.path.join(self.save_path, 'sweeps', 'LIDAR_TOP', filenames[i])
                # print(pc_path)
                pc_raw = np.fromfile(pc_path, dtype = np.float32, count = -1)
                if pc_raw.shape[0] % 5 == 0:
                    pc_raw = pc_raw.reshape([-1, 5])[:, :4]
                else:
                    pc_raw = pc_raw.reshape([-1,4])
                # remove ego car, ground, and max_dist, max_height
                print("before rm", pc_raw.shape[0])
                # rm_min_and_max_dist
                print("----rm_min_and_max_dist----")
                pc_raw[:, :3] = np.dot(pc_raw[:, :3], Quaternion(rot).rotation_matrix)
                for j in range(3):
                    pc_raw[:, j] = pc_raw[:, j] + trl[j]
                dist_origin = np.sqrt(np.sum(pc_raw[:, :3] ** 2, axis = 1))
                keep = np.logical_and(self.min_dist <= dist_origin, dist_origin <= self.max_dist)
                pc_raw = pc_raw[keep]
                print("points_num", pc_raw.shape[0])

                # rm max_height and ground
                print("----rm max_height and ground----")
                # pc_raw = pc_raw[pc_raw[:, 2] > self.rm_thre]
                indices, _ = my_ransac(pc_raw[:, :3])
                print(indices)
                pc_raw[indices] = self.max_height + 1
                pc_raw = pc_raw[pc_raw[:, 2] <= self.max_height]
                print("after rm", pc_raw.shape)
                fn_times.append(filenames[i] + ' ' + str(timestamps[i]))
                
                pc_raw.tofile(data_save_path)
                print(pc_raw.shape)
                num = pc_raw.shape[0]
                if num < self.min_points:
                    self.min_points = num
                if num > self.max_points:
                    self.max_points = num

                max_h = np.max(pc_raw[:, 2])
                min_h = np.min(pc_raw[:, 2])
                print("height", min_h, max_h)
            print(scene_save_path)
            with open(scene_save_path, "w") as f:
                for line in fn_times:
                    f.write(line +'\n')
        print("max_p", self.max_points)
        print("min_p", self.min_points)
        return

class PreprocessKitti:
    def __init__(self, root, save_path, seq, min_dist = 3.0, max_dist = 50.0, max_height = 4.0):
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.max_height = max_height
        self.root = root
        self.save_path = save_path
        self.seq = seq
        self.velodynes = self.read_seq_velodyne() # scene-0001 ...
        self.max_points = 0
        self.min_points = 10**10
        self.max_height = 0
        self.min_height = 0
        self.save_processed()

    def read_seq_velodyne(self):
        # use seq00 for train in general and seq01-10 for validation
        velodyne_path = os.path.join(self.root, "data_odometry_velodyne", "dataset", "sequences", self.seq, "velodyne")
        seq_path_list = glob.glob(os.path.join(velodyne_path, '*.bin'))
        seq_path_list = sorted(seq_path_list)
        return seq_path_list

    def save_processed(self):

        for pc_path in tqdm(self.velodynes):
            print(pc_path)
            if not os.path.exists(os.path.join(self.save_path, self.seq)):
                os.mkdir(os.path.join(self.save_path, self.seq))
            if not os.path.exists(os.path.join(self.save_path, self.seq, "velodyne")):
                os.mkdir(os.path.join(self.save_path, self.seq, "velodyne"))
            save_path = os.path.join(self.save_path, self.seq, "velodyne", pc_path[-10:])
            print(save_path)
            pc_raw = np.fromfile(pc_path, dtype = np.float32, count = -1).reshape([-1, 4])

            # remove ego car, ground, and max_dist, max_height
            print("before rm", pc_raw.shape[0])
            # rm_min_and_max_dist
            print("----rm_min_and_max_dist----")
            dist_origin = np.sqrt(np.sum(pc_raw[:, :3] ** 2, axis = 1))
            keep = np.logical_and(self.min_dist <= dist_origin, dist_origin <= self.max_dist)
            pc_raw = pc_raw[keep]
            print("points_num", pc_raw.shape[0])

            # rm max_height and ground
            print("----rm max_height and ground----")
            indices, _ = my_ransac(pc_raw[:, :3])
            print(indices)
            pc_raw[indices] = self.max_height + 1
            pc_raw = pc_raw[pc_raw[:, 2] <= self.max_height]
            print("after rm", pc_raw.shape)
            
            pc_raw.tofile(save_path)
            num = pc_raw.shape[0]
            if num < self.min_points:
                self.min_points = num
            if num > self.max_points:
                self.max_points = num

            self.max_height = max(np.max(pc_raw[:, 2]), self.max_height)
            self.min_height = min(np.min(pc_raw[:, 2]), self.min_height)

        print("points num: [", self.min_points, ",", self.max_points, "]")
        print("points height: [", self.min_height, ",", self.max_height, "]")
        return


class PreprocessNonlinear:
    def __init__(self, root, scene_dir, save_path, remove = False, min_dist = 3.0, max_dist = 80.0, min_height = -5.0, max_height = 5.0):
        self.root = root
        self.scene_dir = scene_dir
        self.save_path = save_path
        self.remove = remove
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.min_height = min_height
        self.max_height = max_height
        self.velodynes = self.read_sample_velodyne() 
        self.max_for_dataset = 0
        self.max_x = 0
        self.max_y = 0
        self.max_z = 0
        
        
    def read_sample_velodyne(self):
        scene_list = glob.glob(os.path.join(self.scene_dir, '*.txt'))
        scene_list = sorted(scene_list)
        velo_list = []
        for scene in scene_list:
            with open(scene, 'r') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(' ')
                    velo_list.append(line)
        return velo_list # [nsample, 4+interval-1]
    
    def normalize(self, pc, max_for_dataset):
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = max_for_dataset
        pc = pc / m
        return pc

    def rm_range(self, pc_raw, min_dist, max_dist, min_height, max_height):
        dist_origin = np.sqrt(np.sum(pc_raw[:, :3] ** 2, axis = 1))
        keep = np.logical_and(min_dist <= dist_origin, dist_origin <= max_dist)
        pc_raw = pc_raw[keep]
        pc_raw = pc_raw[pc_raw[:, 2] <= max_height]
        pc_raw = pc_raw[pc_raw[:, 2] >= min_height]
        return pc_raw


    def save_processed_kitti(self):
        count = 0
        p = set()
        for i in range(len(self.velodynes)):
            for j in range(len(self.velodynes[0])):
                name = '_'.join([self.velodynes[i][j].split('/')[-3], self.velodynes[i][j].split('/')[-1]])
                p.add(name)
        for i in range(len(self.velodynes)):
            for j in range(len(self.velodynes[0])):
                pc_path = os.path.join(self.root, self.velodynes[i][j])
                print(pc_path)
                name = '_'.join([self.velodynes[i][j].split('/')[-3], self.velodynes[i][j].split('/')[-1]])
                print(name)
                if not os.path.exists(self.save_path):
                    os.mkdir(self.save_path)
                save_path = os.path.join(self.save_path, name)
                print(save_path)

                pc_raw = np.fromfile(pc_path, dtype = np.float32, count = -1).reshape([-1, 4])[:, :3]
                print("==pc_raw==", pc_raw.shape)
                print("pc_raw x range:", np.min(pc_raw[:, 0]), np.max(pc_raw[:, 0]))
                print("pc_raw y range:", np.min(pc_raw[:, 1]), np.max(pc_raw[:, 1]))
                print("pc_raw z range:", np.min(pc_raw[:, 2]), np.max(pc_raw[:, 2]))

                pc_raw = self.rm_range(pc_raw, self.min_dist, self.max_dist, self.min_height, self.max_height)
                pc_raw = np.array(pc_raw, dtype=np.float32)
                print("==max_dist==", self.max_dist, "|| pc_rm:", pc_raw.shape)
                print("pc_rm x range:", np.min(pc_raw[:, 0]), np.max(pc_raw[:, 0]))
                print("pc_rm y range:", np.min(pc_raw[:, 1]), np.max(pc_raw[:, 1]))
                print("pc_rm z range:", np.min(pc_raw[:, 2]), np.max(pc_raw[:, 2]))
                self.max_for_dataset = max(np.max(np.abs(pc_raw)).item(), self.max_for_dataset)
                self.max_x = max(np.max(np.abs(pc_raw[:, 0])).item(), self.max_x)
                self.max_y = max(np.max(np.abs(pc_raw[:, 1])).item(), self.max_y)
                self.max_z = max(np.max(np.abs(pc_raw[:, 2])).item(), self.max_z)

                if self.remove:
                    # rm ground
                    print("----rm ground----")
                    pc_raw = pc_raw[pc_raw[:, 2] >= -1.4]                
                    print("==rm ground==" "pc_rm_ground:", pc_raw.shape)
                    print("pc_rm x range:", np.min(pc_raw[:, 0]), np.max(pc_raw[:, 0]))
                    print("pc_rm y range:", np.min(pc_raw[:, 1]), np.max(pc_raw[:, 1]))
                    print("pc_rm z range:", np.min(pc_raw[:, 2]), np.max(pc_raw[:, 2]))

                # print("pc_raw: ", pc_raw)
                # print("pc_raw x range:", np.min(pc_raw[:, 0]), np.max(pc_raw[:, 0]))
                # print("pc_raw y range:", np.min(pc_raw[:, 1]), np.max(pc_raw[:, 1]))
                # print("pc_raw z range:", np.min(pc_raw[:, 2]), np.max(pc_raw[:, 2]))
                # print(pc_raw.shape)
                # pc_raw = self.normalize(pc_raw, max_for_dataset=81.15216827392578)
                # print("norm_pc_raw: ", pc_raw)
                # print("pc_raw x range:", np.min(pc_raw[:, 0]), np.max(pc_raw[:, 0]))
                # print("pc_raw y range:", np.min(pc_raw[:, 1]), np.max(pc_raw[:, 1]))
                # print("pc_raw z range:", np.min(pc_raw[:, 2]), np.max(pc_raw[:, 2]))
                if not os.path.isfile(save_path):
                    count += 1
                    pc_raw.tofile(save_path)
        
        names_txt_path = os.path.join(self.save_path, "scene_list")
        if not os.path.exists(names_txt_path):
            os.mkdir(names_txt_path)
        with open(os.path.join(names_txt_path, "scene_list.txt"), 'a') as f:
            for i in range(len(self.velodynes)):
                for j in range(len(self.velodynes[0])-1):
                    name = '_'.join([self.velodynes[i][j].split('/')[-3], self.velodynes[i][j].split('/')[-1]])
                    f.write(name + ' ')
                name = '_'.join([self.velodynes[i][-1].split('/')[-3], self.velodynes[i][-1].split('/')[-1]])
                f.write(name + '\n')
        print("====dataset range====")
        print("max_for_kitti: ", self.max_for_dataset)
        print("max_x for kitti:", self.max_x)
        print("max_y for kitti:", self.max_y)
        print("max_z for kitti:", self.max_z)
        print("file to write: ", count)
        print("file should be written: ", len(p))
        assert len(p) == count


    def save_processed_nuscenes(self):
        count = 0
        p = set()
        for i in range(len(self.velodynes)):
            for j in range(len(self.velodynes[0])):
                name = self.velodynes[i][j].split('/')[-1]
                p.add(name)
        for i in range(len(self.velodynes)):
            for j in range(len(self.velodynes[0])):
                pc_path = os.path.join(self.root, self.velodynes[i][j])
                print(pc_path)
                name = self.velodynes[i][j].split('/')[-1]
                save_path = os.path.join(self.save_path, name)
                print(save_path)
                pc_raw = np.fromfile(pc_path, dtype = np.float32, count = -1)
                if not pc_raw.shape[0] % 5:
                    pc_raw = pc_raw.reshape(-1, 5)[:, :3]
                else:
                    pc_raw = pc_raw.reshape(-1, 4)[:, :3]
                # print(pc_raw.shape)
                pc_raw = np.array(pc_raw, dtype=np.float32)
                print("==pc_raw==", pc_raw.shape)
                print("pc_raw x range:", np.min(pc_raw[:, 0]), np.max(pc_raw[:, 0]))
                print("pc_raw y range:", np.min(pc_raw[:, 1]), np.max(pc_raw[:, 1]))
                print("pc_raw z range:", np.min(pc_raw[:, 2]), np.max(pc_raw[:, 2]))

                pc_raw = self.rm_range(pc_raw, self.min_dist, self.max_dist, self.min_height, self.max_height)
                print("==max_dist==", self.max_dist, "|| pc_rm:", pc_raw.shape)
                print("pc_rm x range:", np.min(pc_raw[:, 0]), np.max(pc_raw[:, 0]))
                print("pc_rm y range:", np.min(pc_raw[:, 1]), np.max(pc_raw[:, 1]))
                print("pc_rm z range:", np.min(pc_raw[:, 2]), np.max(pc_raw[:, 2]))
                self.max_for_dataset = max(np.max(np.abs(pc_raw)).item(), self.max_for_dataset)
                self.max_x = max(np.max(np.abs(pc_raw[:, 0])).item(), self.max_x)
                self.max_y = max(np.max(np.abs(pc_raw[:, 1])).item(), self.max_y)
                self.max_z = max(np.max(np.abs(pc_raw[:, 2])).item(), self.max_z)


                if self.remove:
                    # rm ground
                    print("----rm ground----")
                    indices, _ = my_ransac(pc_raw)
                    pc_raw[indices] = self.max_height + 1
                    pc_raw = pc_raw[pc_raw[:, 2] <= self.max_height]                
                    print("==rm ground==" "pc_rm_ground:", pc_raw.shape)
                    print("pc_rm x range:", np.min(pc_raw[:, 0]), np.max(pc_raw[:, 0]))
                    print("pc_rm y range:", np.min(pc_raw[:, 1]), np.max(pc_raw[:, 1]))
                    print("pc_rm z range:", np.min(pc_raw[:, 2]), np.max(pc_raw[:, 2]))

                # print("pc_raw: ", pc_raw)
                # print("pc_raw x range:", np.min(pc_raw[:, 0]), np.max(pc_raw[:, 0]))
                # print("pc_raw y range:", np.min(pc_raw[:, 1]), np.max(pc_raw[:, 1]))
                # print("pc_raw z range:", np.min(pc_raw[:, 2]), np.max(pc_raw[:, 2]))
                # print(pc_raw.shape)
                # pc_raw = self.normalize(pc_raw, max_for_dataset = 105.2951431274414)
                # print("norm_pc_raw: ", pc_raw)
                # print("pc_raw x range:", np.min(pc_raw[:, 0]), np.max(pc_raw[:, 0]))
                # print("pc_raw y range:", np.min(pc_raw[:, 1]), np.max(pc_raw[:, 1]))
                # print("pc_raw z range:", np.min(pc_raw[:, 2]), np.max(pc_raw[:, 2]))
                if not os.path.isfile(save_path):
                    count += 1
                    pc_raw.tofile(save_path)
        
        names_txt_path = os.path.join(self.save_path, "scene_list")
        if not os.path.exists(names_txt_path):
            os.mkdir(names_txt_path)
        with open(os.path.join(names_txt_path, "scene_list.txt"), 'a') as f:
            for i in range(len(self.velodynes)):
                for j in range(len(self.velodynes[0]) - 1):
                    name = self.velodynes[i][j].split('/')[-1]
                    f.write(name + ' ')
                name = self.velodynes[i][-1].split('/')[-1]
                f.write(name + '\n')
        print("max_for_nscs: ", self.max_for_dataset)
        print("max_x for nscs:", self.max_x)
        print("max_y for nscs:", self.max_y)
        print("max_z for nscs:", self.max_z)
        print("file to write: ", count)
        print("file should be written:", len(p))
        assert len(p) == count
        

    def save_processed_argos(self):
        count = 0
        p = set()
        for i in range(len(self.velodynes)):
            for j in range(len(self.velodynes[0])):
                name = self.velodynes[i][j].split('/')[-1]
                name = name.split('.')[0]
                p.add(name)
        for i in range(len(self.velodynes)):
            for j in range(len(self.velodynes[0])):
                pc_path = os.path.join(self.root, self.velodynes[i][j])
                print(pc_path)
                name = self.velodynes[i][j].split('/')[-1]
                name = name.split('.')[0]
                save_path = os.path.join(self.save_path, name + '.bin')
                print(save_path)
                pc_raw = io_utils.read_lidar_sweep(pc_path, attrib_spec="xyz")
                # print("pc_raw: ", pc_raw)
                # print("pc_raw x range:", np.min(pc_raw[:, 0]), np.max(pc_raw[:, 0]))
                # print("pc_raw y range:", np.min(pc_raw[:, 1]), np.max(pc_raw[:, 1]))
                # print("pc_raw z range:", np.min(pc_raw[:, 2]), np.max(pc_raw[:, 2]))
                pc_raw = pc_raw.reshape(-1, 3)
                # ar = np.ones((pc_raw.shape[0], 1))
                # pc_raw = np.hstack((pc_raw, ar))                
                pc_raw = np.array(pc_raw, dtype=np.float32)
                print("==pc_raw==", pc_raw.shape)
                print("pc_raw x range:", np.min(pc_raw[:, 0]), np.max(pc_raw[:, 0]))
                print("pc_raw y range:", np.min(pc_raw[:, 1]), np.max(pc_raw[:, 1]))
                print("pc_raw z range:", np.min(pc_raw[:, 2]), np.max(pc_raw[:, 2]))
              
                pc_raw = self.rm_range(pc_raw, self.min_dist, self.max_dist, self.min_height, self.max_height)
                pc_raw = np.array(pc_raw, dtype=np.float32)
                print("==max_dist==", self.max_dist, "|| pc_rm:", pc_raw.shape)
                print("pc_rm x range:", np.min(pc_raw[:, 0]), np.max(pc_raw[:, 0]))
                print("pc_rm y range:", np.min(pc_raw[:, 1]), np.max(pc_raw[:, 1]))
                print("pc_rm z range:", np.min(pc_raw[:, 2]), np.max(pc_raw[:, 2]))
                self.max_for_dataset = max(np.max(np.abs(pc_raw)).item(), self.max_for_dataset)
                self.max_x = max(np.max(np.abs(pc_raw[:, 0])).item(), self.max_x)
                self.max_y = max(np.max(np.abs(pc_raw[:, 1])).item(), self.max_y)
                self.max_z = max(np.max(np.abs(pc_raw[:, 2])).item(), self.max_z)

                if self.remove:
                    # rm ground
                    print("----rm ground----")
                    indices, _ = my_ransac(pc_raw)
                    pc_raw[indices] = self.max_height + 1
                    pc_raw = pc_raw[pc_raw[:, 2] <= self.max_height]                
                    print("==rm ground==" "pc_rm_ground:", pc_raw.shape)
                    print("pc_rm x range:", np.min(pc_raw[:, 0]), np.max(pc_raw[:, 0]))
                    print("pc_rm y range:", np.min(pc_raw[:, 1]), np.max(pc_raw[:, 1]))
                    print("pc_rm z range:", np.min(pc_raw[:, 2]), np.max(pc_raw[:, 2]))

                # print("pc_raw: ", pc_raw)
                # print("pc_raw x range:", np.min(pc_raw[:, 0]), np.max(pc_raw[:, 0]))
                # print("pc_raw y range:", np.min(pc_raw[:, 1]), np.max(pc_raw[:, 1]))
                # print("pc_raw z range:", np.min(pc_raw[:, 2]), np.max(pc_raw[:, 2]))
                # print("pc_raw i range:", np.min(pc_raw[:, 3]), np.max(pc_raw[:, 3]))
                # print(pc_raw.shape)
                # pc_raw = self.normalize(pc_raw, max_for_dataset=228.75)
                # print("norm_pc_raw: ", pc_raw)
                # print("pc_raw x range:", np.min(pc_raw[:, 0]), np.max(pc_raw[:, 0]))
                # print("pc_raw y range:", np.min(pc_raw[:, 1]), np.max(pc_raw[:, 1]))
                # print("pc_raw z range:", np.min(pc_raw[:, 2]), np.max(pc_raw[:, 2]))
                # print(pc_raw.shape)
                if not os.path.isfile(save_path):
                    count += 1
                    pc_raw.tofile(save_path)
        
        names_txt_path = os.path.join(self.save_path, "scene_list")
        if not os.path.exists(names_txt_path):
            os.mkdir(names_txt_path)
        with open(os.path.join(names_txt_path, "scene_list.txt"), 'a') as f:
            for i in range(len(self.velodynes)):
                for j in range(len(self.velodynes[0])-1):
                    name = self.velodynes[i][j].split('/')[-1]
                    name = name.split('.')[0]
                    f.write(name + '.bin ')
                name = self.velodynes[i][-1].split('/')[-1]
                name = name.split('.')[0]
                f.write(name + '.bin' + '\n')
        print("max_for_argos: ", self.max_for_dataset)
        print("max_x for argos:", self.max_x)
        print("max_y for argos:", self.max_y)
        print("max_z for argos:", self.max_z)
        print("file to write: ", count)
        print("file should be written:", len(p))
        assert len(p) == count



if __name__ == "__main__":
    args = parse_arg()
    # processor = PreprocessNuscenes(args.root, args.save_path, args.scenes_list, args.min_dist, args.max_dist, args.max_height)
    # processor = PreprocessKitti(args.root, args.save_path, "00")
    
    # this part is to deal with kitti subdataset
    if args.dataset == 'kitti':
        processor = PreprocessNonlinear(args.root, args.scenes_list ,args.save_path, remove = args.remove)
        processor.save_processed_kitti()

    # this part is to deal with nuscenes subdataset
    elif args.dataset == 'nscs':
        processor = PreprocessNonlinear(args.root, args.scenes_list ,args.save_path, remove = args.remove)
        processor.save_processed_nuscenes()

    # this part is to deal with argos subdataset
    elif args.dataset == 'argos':
        processor = PreprocessNonlinear(args.root, args.scenes_list ,args.save_path, remove = args.remove)
        processor.save_processed_argos()