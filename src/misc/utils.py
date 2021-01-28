import glob
import os
import json
import operator
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_best_checkpoint(path, compare_value='epoch_num', comparator='max'):
    with open(os.path.join(path, "stats.json"), "r") as read_file:
        data = json.load(read_file)
        checkpoints = [epoch_stat[compare_value] for epoch_stat in data]
        if comparator is 'max':
            chckp = max((checkpoints, i) for i, checkpoints in enumerate(checkpoints))
        elif comparator is 'min':
            chckp = min((checkpoints, i) for i, checkpoints in enumerate(checkpoints))
        return (chckp, data)
        
def get_best_chkpts(path, metric_name, comparator='>'):
    """
    Return the best checkpoint according to some criteria.
    Note that it will only return valid path, so any checkpoint that has been
    removed wont be returned (i.e moving to next one that satisfies the criteria
    such as second best etc.)

    Args:
        path: directory contains all checkpoints, including the "stats.json" file
    """
    # info = []
    # for stat_file in glob.glob(f"{path}/*/*.json"):
    #     print (stat_file)
    #     stat = json.load(stat_file)
    #     info.append(stat)
    # print (info)

    stat_file = os.path.join(path, 'stats.json')
    with open(stat_file) as f:
        info = json.load(f)

    ops = {
            '>': operator.gt,
            '<': operator.lt,
          }
    op_func = ops[comparator]
    
    if comparator == '>':
        best_value  = -float("inf")
    else:
        best_value  = +float("inf")

    best_chkpt = None
    for epoch_stat in info:
        epoch_value = epoch_stat[metric_name]
        if op_func(epoch_value, best_value):
            chkpt_path = os.path.join(path, 'model-{}.index'.format(epoch_stat['global_step']))
            if os.path.isfile(chkpt_path):
                selected_stat = epoch_stat
                best_value  = epoch_value
                best_chkpt = chkpt_path
    return best_chkpt, selected_stat

####
def normalize(mask, dtype=np.uint8):
    return (255 * mask / np.amax(mask)).astype(dtype)

####
def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]

####
def cropping_center(x, crop_shape, batch=False):
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0:h0 + crop_shape[0], w0:w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:,h0:h0 + crop_shape[0], w0:w0 + crop_shape[1]]
    return x

####
def rm_n_mkdir(dir_path):
    if (os.path.isdir(dir_path)):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

####
def get_files(data_dir_list, data_ext):
    """
    Given a list of directories containing data with extention 'data_ext',
    generate a list of paths for all files within these directories
    """
    data_files = []
    for sub_dir in data_dir_list:
        files_list = glob.glob('{}/*{}'.format(sub_dir, data_ext))
        files_list.sort() # ensure same order
        data_files.extend(files_list)
    return data_files

####
def get_inst_centroid(inst_map):
    inst_centroid_list = []
    inst_id_list = list(np.unique(inst_map))
    for inst_id in inst_id_list[1:]: # avoid 0 i.e background
        mask = np.array(inst_map == inst_id, np.uint8)
        inst_moment = cv2.moments(mask)
        inst_centroid = [(inst_moment["m10"] / inst_moment["m00"]),
                         (inst_moment["m01"] / inst_moment["m00"])]
        inst_centroid_list.append(inst_centroid)
    return np.array(inst_centroid_list)

####
def show_np_array(array):
    plt.imshow(array)
    plt.show()
    plt.pause(5)
    plt.close()
