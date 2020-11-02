import os
import numpy as np
from math import radians, tan, sqrt
import cv2
from params import par
import time
import multiprocessing as mp
from functools import partial

velo_dir = '/data/KITTI_VO_Benchmark/dataset/sequences/'
img_path = '/data/000001.png'
def create_velo_data():
    info = {'00': [0, 4540], '01': [0, 1100], '02': [0, 4660], '03': [0, 800], '04': [0, 270], '05': [0, 2760], '06': [0, 1100], '07': [0, 1100], '08': [1100, 5170], '09': [0, 1590], '10': [0, 1200]}

    for video in info.keys():
        # TODO: mkdir for each video
        dir_path ='/home/dl/DeepVO-pytorch/KITTI/velodynes/{}'.format(video)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        else:
            continue
    # TODO: read each frame
    for video in info.keys():
        start_t = time.time()
        fn = '{}{}/velodyne/'.format(velo_dir, video) 
        velo_files = os.listdir(fn)
        print('Transforming {}...'.format(fn))
        # multiprocessing
        part_process = partial(process_velo_scans, fn=fn, video=video)
        pool = mp.Pool()
        pool.map(part_process, velo_files)
        # for velo_file in velo_files:
        #     velo_scans = load_velo_scan(fn+velo_file)
        #     horizontal_scans = np.array(extract_horizontal_layer(velo_scans),dtype=np.float32)
        #     # print('-'*50)
        #     # print('%d points of the horizontal layer were extracted!' % len(horizontal_scans))
        #     # print('-'*50)
        #     img = polar_registration(horizontal_scans)
        #     base_fn = os.path.splitext(velo_file)[0]
        #     np.save('/home/dl/DeepVO-pytorch/KITTI/velodynes/{}/'.format(video)+base_fn+'.npy', img)
        #     print('Video {}-{}: shape={}'.format(video,velo_file, img.shape))
        print('elapsed time = {}'.format(time.time()-start_t))


def load_velo_scan(file):
    """Load and parse a velodyne binary file."""
    scan = np.fromfile(file, dtype=np.float32)
    return scan.reshape((-1, 4))

def process_velo_scans(velo_files, fn, video):
    """Generator to parse velodyne binary files into npy arrays."""
    velo_file = velo_files
    velo_scans = load_velo_scan(fn+velo_file)
    horizontal_scans = np.array(extract_horizontal_layer(velo_scans),dtype=np.float32)
    # print('-'*50)
    # print('%d points of the horizontal layer were extracted!' % len(horizontal_scans))
    # print('-'*50)
    img = polar_registration(horizontal_scans)
    base_fn = os.path.splitext(velo_file)[0]
    np.save('/home/dl/DeepVO-pytorch/KITTI/velodynes/{}/'.format(video)+base_fn+'.npy', img)
    # print('Video {}-{}: shape={}'.format(video,velo_file, img.shape))

def extract_horizontal_layer(scans, verti_res=0.4):
    """Extract scans in the horizontal layer."""
    horizontal_layer = []
    # vertical resolution of Velodyne_HDL-64E is 0.4 degree
    max_vertical_drift = tan(radians(verti_res/2))
    for scan in scans:
        # set bounds of horizontal layer
        if abs(scan[2])/sqrt(scan[0]**2 + scan[1]**2) < max_vertical_drift:
            horizontal_layer.append(scan)
    return horizontal_layer


def polar_registration(scans, hori_res=0.4):
    """Register scans in a discrete polar coordinate system with a specific resolution."""
    polar_map = [[] for i in range(int(360/hori_res+1))]
    for scan in scans:
        r = sqrt(scan[0]**2 + scan[1]**2)
        theta = cv2.fastAtan2(scan[1], scan[0])
        polar_idx = int(theta / hori_res)
        polar_map[polar_idx].append(r)
    for j,beam in enumerate(polar_map):
        if beam:
            polar_map[j] = np.mean(beam)
        else:
            polar_map[j] = 0.
    return np.array(polar_map, dtype=float)




def pixel_registration(scans, height=1600, width=1200, range_x=160, range_y=120, show_vehicle=False):
    scale_x = height/range_x
    scale_y = width/range_y
    img = np.zeros((height, width))
    for scan in scans:
        scan[0] = scale_x * scan[0] + height/2
        scan[1] = scale_y * scan[1] + width/2
        scan[3] = 255 * scan[3]
        img[int(scan[0]),int(scan[1])] = int(scan[3])
        # show vehicle position
        if show_vehicle:
            img[790:810,590:610] = 255
    return img

# # 封装
# # def img_write()
# velo_scans = load_velo_scan('/data/KITTI_VO_Benchmark/dataset/sequences/00/velodyne/000000.bin')
# horizontal_scans = np.array(extract_horizontal_layer(velo_scans),dtype=np.float32)
# print('-'*50)
# print('%d points of the horizontal layer were extracted!' % len(horizontal_scans))
# print('-'*50)
# img = polar_registration(horizontal_scans)
# print(img)
# # cv2.imwrite(img_path, img)
if __name__=='__main__':
    create_velo_data()