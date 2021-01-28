import glob
import os
import argparse

import cv2
import numpy as np

from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir

from config import Config


if __name__ == '__main__':
    cfg = Config()
    
    normalized = False
    for mode in cfg.data_modes:
        if cfg.out_preproc is not None:
            normalized = (os.path.exists(cfg.out_preproc[mode]) and \
                len(os.listdir(cfg.out_preproc[mode])) != 0)
        if not normalized: 
            normalized = False
            break
        
    assert (cfg.normalized == normalized)
    print(f"Stain normalization was performed: {normalized}")
    img_dirs = cfg.out_preproc if normalized else cfg.img_dirs

    print(f"Using folders <{list(img_dirs.values())}> as input")
    print(f"Saving results to <{list(cfg.out_extract.values())}>")

    for data_mode in img_dirs.keys():
        xtractor = PatchExtractor(cfg.win_size, cfg.step_size)

        img_dir = img_dirs[data_mode]
        ann_dir = cfg.labels_dirs[data_mode]
        
        file_list = glob.glob(os.path.join(img_dir, '*{}'.format(cfg.img_ext)))
        file_list.sort()
        out_dir = cfg.out_extract[data_mode]

        rm_n_mkdir(out_dir)
        for filename in file_list:
            filename = os.path.basename(filename)
            basename = filename.split('.')[0]
            print('Mode: {}, filename - {}'.format(data_mode, filename))

            img = cv2.imread(os.path.join(img_dir, '{}{}'.format(basename, cfg.img_ext)))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if cfg.type_classification:
                # # assumes that ann is HxWx2 (nuclei class labels are available at index 1 of C)
                ann = np.load(os.path.join(ann_dir, '{}.npy'.format(basename)), allow_pickle=True)
                # ann_inst = ann[...,0]
                # ann_type = ann[...,1]
                ann_inst = ann.item().get('inst_map')
                ann_type = ann.item().get('type_map')

                #ann = sio.loadmat(os.path.join(ann_dir, '{}.mat'.format(basename)))
                #ann_inst = ann['inst_map']
                #ann_type = ann['type_map']

                # merge classes for CoNSeP (in paper we only utilise 3 nuclei classes and background)
                # If own dataset is used, then the below may need to be modified
                # TODO: move to preproc CoNSeP dataset
                # ann_type[(ann_type == 3) | (ann_type == 4)] = 3
                # ann_type[(ann_type == 5) | (ann_type == 6) | (ann_type == 7)] = 4
                
                # print (f"nr_types = {cfg.nr_types}, max in annotation = {np.max(ann_type)}")
                assert np.max(ann_type) <= cfg.nr_types, \
                                ("Only {} types of nuclei are defined for training"\
                                "but there are {} types found in the input image.").format(cfg.nr_types, np.max(ann_type))

                ann = np.dstack([ann_inst, ann_type])
                ann = ann.astype('int32')
            else:
                # assumes that ann is HxW
                # ann_inst = sio.loadmat(os.path.join(ann_dir, '{}.mat'.format(basename)))
                ann = np.load(os.path.join(ann_dir, '{}.npy'.format(basename)))
                ann_inst = (ann_inst.item().get('inst_map')).astype('int32')
                # ann_inst = ann_inst.astype('int32')
                ann = np.expand_dims(ann_inst, -1)
                
            img = np.concatenate([img, ann], axis=-1)
            sub_patches = xtractor.extract(img, cfg.extract_type)
            for idx, patch in enumerate(sub_patches):
                np.save("{0}/{1}_{2:03d}.npy".format(out_dir, basename, idx), patch)
            print (f"{out_dir}/{basename} saved.")
