import argparse
import glob
import math
import os
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import scipy.io as sio
from scipy import io as sio
from scipy.ndimage import filters, measurements
from scipy.ndimage.morphology import (binary_dilation, binary_fill_holes,
                                      distance_transform_cdt,
                                      distance_transform_edt)
from skimage.morphology import remove_small_objects, watershed
from tensorpack.predict import OfflinePredictor

import postproc.hover
from config import Config
from metrics.stats_utils import remap_label
from misc.utils import get_inst_centroid, rm_n_mkdir
from misc.viz_utils import visualize_instances


def swap_classes(pred, mapping):
    '''
    # Example: change 1 to 4 and 2 to 3.
    mapping = {1:4, 2:3}
    '''
    id_map = pred.copy()
    for k,v in mapping.items():
        pred[id_map == k] = v
    return pred

####
class Inferer(Config):
    def __gen_prediction(self, x, predictor):
        """
        Using 'predictor' to generate the prediction of image 'x'

        Args:
            x : input image to be segmented. It will be split into patches
                to run the prediction upon before being assembled back
        """
        step_size = self.infer_mask_shape
        msk_size = self.infer_mask_shape
        win_size = self.infer_input_shape

        def get_last_steps(length, msk_size, step_size):
            nr_step = math.ceil((length - msk_size) / step_size)
            last_step = (nr_step + 1) * step_size
            return int(last_step), int(nr_step + 1)

        im_h = x.shape[0]
        im_w = x.shape[1]

        last_h, nr_step_h = get_last_steps(im_h, msk_size[0], step_size[0])
        last_w, nr_step_w = get_last_steps(im_w, msk_size[1], step_size[1])

        diff_h = win_size[0] - step_size[0]
        padt = diff_h // 2
        padb = last_h + win_size[0] - im_h

        diff_w = win_size[1] - step_size[1]
        padl = diff_w // 2
        padr = last_w + win_size[1] - im_w

        x = np.lib.pad(x, ((padt, padb), (padl, padr), (0, 0)), "reflect")

        #### TODO: optimize this
        sub_patches = []
        # generating subpatches from orginal
        for row in range(0, last_h, step_size[0]):
            for col in range(0, last_w, step_size[1]):
                win = x[row : row + win_size[0], col : col + win_size[1]]
                sub_patches.append(win)

        pred_map = deque()
        while len(sub_patches) > self.inf_batch_size:
            mini_batch = sub_patches[: self.inf_batch_size]
            sub_patches = sub_patches[self.inf_batch_size :]
            mini_output = predictor(mini_batch)[0]
            mini_output = np.split(mini_output, self.inf_batch_size, axis=0)
            pred_map.extend(mini_output)
        if len(sub_patches) != 0:
            mini_output = predictor(sub_patches)[0]
            mini_output = np.split(mini_output, len(sub_patches), axis=0)
            pred_map.extend(mini_output)

        #### Assemble back into full image
        output_patch_shape = np.squeeze(pred_map[0]).shape
        ch = 1 if len(output_patch_shape) == 2 else output_patch_shape[-1]

        #### Assemble back into full image
        pred_map = np.squeeze(np.array(pred_map))
        pred_map = np.reshape(pred_map, (nr_step_h, nr_step_w) + pred_map.shape[1:])
        pred_map = (
            np.transpose(pred_map, [0, 2, 1, 3, 4])
            if ch != 1
            else np.transpose(pred_map, [0, 2, 1, 3])
        )
        pred_map = np.reshape(
            pred_map,
            (
                pred_map.shape[0] * pred_map.shape[1],
                pred_map.shape[2] * pred_map.shape[3],
                ch,
            ),
        )
        pred_map = np.squeeze(pred_map[:im_h, :im_w])  # just crop back to original size

        return pred_map

    ####
    def run(self):
        cfg = Config()
        energy_mode = 2
        marker_mode = 2
        proc_dir = os.path.join(cfg.inf_output_dir, 'processed')
        if not os.path.isdir(proc_dir):
            os.makedirs(proc_dir)

        #### function
        def process_image(img, pred):
            pred_inst = pred[..., cfg.nr_types :]
            pred_type = pred[..., : cfg.nr_types]

            pred_inst = np.squeeze(pred_inst)
            pred_type = np.argmax(pred_type, axis=-1)

            if cfg.model_type == "np_hv" or cfg.model_type == "np_hv_opt":
                pred_inst = postproc.hover.proc_np_hv(
                    pred_inst, marker_mode=marker_mode, energy_mode=energy_mode, rgb=img
                )

            # ! will be extremely slow on WSI/TMA so it's advisable to comment this out
            # * remap once so that further processing faster (metrics calculation, etc.)
            if cfg.remap_labels:
                pred_inst = remap_label(pred_inst, by_size=True)

            #### * Get class of each instance id, stored at index id-1
            pred_id_list = list(np.unique(pred_inst))[1:]  # exclude background ID
            pred_inst_type = np.full(len(pred_id_list), 0, dtype=np.int32)
            for idx, inst_id in enumerate(pred_id_list):
                inst_type = pred_type[pred_inst == inst_id]
                type_list, type_pixels = np.unique(inst_type, return_counts=True)
                type_list = list(zip(type_list, type_pixels))
                type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
                inst_type = type_list[0][0]
                if inst_type == 0:  # ! pick the 2nd most dominant if exist
                    if len(type_list) > 1:
                        inst_type = type_list[1][0]
                    else:
                        pass  # print('[Warn] Instance has `background` type')
                pred_inst_type[idx] = inst_type
            pred_inst_centroid = get_inst_centroid(pred_inst)


            if cfg.process_mapping is not None:
                pred_type = swap_classes(pred_type, cfg.process_mapping)
                pred_inst_type = swap_classes(pred_inst_type, cfg.process_mapping)

            ###### ad hoc for pannuke predictions
            # 5 -> 4

            ###### ad hoc for squash_monusac predictions
            # 3 -> 2
            # 4 -> 2

            ###### ad hoc for consep model to monusac data
            # 3 -> 2
            # 4 -> 2
            # 1 -> 3

            ###### ad hoc for monusac to consep predictions
            # 3 -> 2
            # 4 -> 2
            # 1 -> 3

            ###### ad hoc for monusac to pannuke predictions
            # 1 -> 4
            # 2 -> 1
            # 3 -> 1
            # 4 -> 1
            # 5 -> 4

            # print (np.unique(pred_type))
            # if 0 not in np.unique(pred_type):
            #     return

            overlaid_output = visualize_instances(
                pred_inst,
                img,
                ((cfg.nuclei_type_dict, cfg.color_palete), pred_inst_type[:, None]),
                cfg.outline,
                cfg.skip_types,
            )
            overlaid_output = cv2.cvtColor(overlaid_output, cv2.COLOR_BGR2RGB)
            cv2.imwrite(
                os.path.join(proc_dir, "{}.png".format(basename)), overlaid_output
            )
            with open(os.path.join(proc_dir, f"{basename}.log"), "w") as log_file:
                unique, counts = np.unique(
                    pred_inst_type[:, None], return_counts=True
                )
                unique = list(unique)
                if 0 in unique:  # remove backround entries
                    counts = np.delete(counts, unique.index(0))
                    unique.remove(0)
                print(
                    f"{basename} : {dict(zip([{str(v): str(k) for k, v in cfg.nuclei_type_dict.items()}[str(item)] for item in unique], counts))}",
                    file=log_file,
                )

        ####

        predictor = OfflinePredictor(self.gen_pred_config())
        for num, data_dir in enumerate(self.inf_data_list):
            file_list = glob.glob(
                os.path.join(data_dir, "*{}".format(self.inf_imgs_ext))
            )
            file_list.sort()  # ensure same order

            # rm_n_mkdir(save_dir)
            for filename in file_list:
                filename = os.path.basename(filename)
                basename = filename.split(".")[0]
                print(data_dir, basename, end=" ", flush=True)

                img = cv2.imread(os.path.join(data_dir, filename))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pred = np.squeeze(self.__gen_prediction(img, predictor))

                process_image(img, pred)
                print(f" - {datetime.now().strftime('%H:%M:%S.%f')} - Finished {filename}")


####
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", help="Comma separated list of GPU(s) to use.", default="0"
    )
    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    n_gpus = len(args.gpu.split(","))

    inferer = Inferer()
    inferer.run()
