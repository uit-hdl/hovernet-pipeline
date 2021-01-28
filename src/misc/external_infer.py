import argparse
import math
import os
from datetime import datetime
from collections import deque
import importlib

import cv2
import numpy as np
from scipy import io as sio

import tensorflow as tf
from tensorpack.predict import OfflinePredictor, PredictConfig
from tensorpack.tfutils.sessinit import SaverRestoreRelaxed


class InfererExternal():
    '''
    Example using tensorpack:
    python external_infer.py \
        --model_path /data/output/export/model/serving/variables/variables.data-00000-of-00001 \
        --input_img /data/output/image.png \
        --save_dir /data/output/

    Example with compact tf model:
    python external_infer.py \
        --model_path /data/output/export/model/compact.pb \
        --input_img /data/output/image.png \
        --save_dir /data/output/
    '''


    def __init__(self, model_path, input_img, save_dir):
        # values for np_hv model graph
        self.infer_mask_shape = [80,  80] # [164, 164]
        self.infer_input_shape = [270, 270] # [256, 256]
        self.inf_batch_size = 16
        self.eval_inf_input_tensor_names = ['images:0']
        self.eval_inf_output_tensor_names = ['predmap-coded:0']

        self.inf_model_path = model_path # NOT np or npz weights like infer.py uses
        self.save_dir = save_dir
        self.input_img_path = input_img


    def __gen_prediction(self, x, predictor, compact=None):

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

        x = np.lib.pad(x, ((padt, padb), (padl, padr), (0, 0)), 'reflect')
        #### TODO: optimize this
        sub_patches = []
        # generating subpatches from orginal
        for row in range(0, last_h, step_size[0]):
            for col in range (0, last_w, step_size[1]):
                win = x[row:row+win_size[0],
                        col:col+win_size[1]]
                sub_patches.append(win)
        pred_map = deque()
        while len(sub_patches) > self.inf_batch_size:
            mini_batch  = sub_patches[:self.inf_batch_size]
            sub_patches = sub_patches[self.inf_batch_size:]
            mini_output = predictor(mini_batch)[0] if compact==False else predictor(mini_batch)
            mini_output = np.split(mini_output, self.inf_batch_size, axis=0)
            pred_map.extend(mini_output)
        if len(sub_patches) != 0:
            mini_output = predictor(sub_patches)[0] if compact==False else predictor(sub_patches)
            mini_output = np.split(mini_output, len(sub_patches), axis=0)
            pred_map.extend(mini_output)

        #### Assemble back into full image
        output_patch_shape = np.squeeze(pred_map[0]).shape
        ch = 1 if len(output_patch_shape) == 2 else output_patch_shape[-1]

        #### Assemble back into full image
        pred_map = np.squeeze(np.array(pred_map))
        pred_map = np.reshape(pred_map, (nr_step_h, nr_step_w) + pred_map.shape[1:])
        pred_map = np.transpose(pred_map, [0, 2, 1, 3, 4]) if ch != 1 else \
                        np.transpose(pred_map, [0, 2, 1, 3])
        pred_map = np.reshape(pred_map, (pred_map.shape[0] * pred_map.shape[1],
                                         pred_map.shape[2] * pred_map.shape[3], ch))
        pred_map = np.squeeze(pred_map[:im_h,:im_w]) # just crop back to original size

        return pred_map


    def apply_compact(self, prefix='import/'):
        """Run the pruned and frozen inference graph.
        TODO: Should be implemented like TF serving without tensorpack.
        """
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            with tf.gfile.GFile(self.inf_model_path, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def)

                input_img = sess.graph.get_tensor_by_name('{}{}'.format(prefix, self.eval_inf_input_tensor_names[0]))
                prediction_img = sess.graph.get_tensor_by_name('{}{}'.format(prefix, self.eval_inf_output_tensor_names[0]))
                img = cv2.cvtColor(cv2.imread(self.input_img_path), cv2.COLOR_BGR2RGB)
                basename = os.path.basename(self.input_img_path).split('.')[0]
                ###
                def predictor_gen():
                    def predictor(image):
                        image = np.array(image)
                        return sess.run(prediction_img, {input_img: image})
                    return predictor
                ###
                pred_map = self.__gen_prediction(img, predictor_gen(), compact=True)
                sio.savemat(os.path.join(self.save_dir,'{}.mat'.format(basename)), {'result':[pred_map]})
                print(f"Finished. {datetime.now().strftime('%H:%M:%S.%f')}")


    def apply_model(self):
        # Using tensorpack!! predictor
        model_constructor = importlib.import_module('model.graph')
        model_constructor = model_constructor.Model_NP_HV

        pred_config = PredictConfig(
            session_init=SaverRestoreRelaxed(self.inf_model_path),
            model=model_constructor(),
            input_names=self.eval_inf_input_tensor_names,
            output_names=self.eval_inf_output_tensor_names)
        ###
        predictor = OfflinePredictor(pred_config)
        img = cv2.cvtColor(cv2.imread(self.input_img_path), cv2.COLOR_BGR2RGB)
        basename = os.path.basename(self.input_img_path).split('.')[0]
        ###
        pred_map = self.__gen_prediction(img, predictor, compact=False)
        sio.savemat(os.path.join(self.save_dir,'{}.mat'.format(basename)), {'result':[pred_map]})
        print(f"Finished. {datetime.now().strftime('%H:%M:%S.%f')}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='Comma separated list of GPU(s) to use.', default="0")
    parser.add_argument('--model_path', help='Path to the model pb/checkpoint', required=True)
    parser.add_argument('--input_img', help='Full path to input image', required=True)
    parser.add_argument('--save_dir', help='Path to the directory to save .mat result', required=True)
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpus = len(args.gpu.split(','))

    inferer = InfererExternal(args.model_path, args.input_img, args.save_dir)
    if (args.model_path.endswith('.pb')): inferer.apply_compact()
    else: inferer.run()