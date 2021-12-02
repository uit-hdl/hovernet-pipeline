import os
import sys
import importlib
import random
import yaml
from collections import defaultdict

import cv2
import numpy as np
import tensorflow as tf
from tensorpack import imgaug, PredictConfig, SaverRestoreRelaxed
from tensorpack.tfutils import get_model_loader

from loader.augs import (
    BinarizeLabel,
    GaussianBlur,
    GenInstanceDistance,
    GenInstanceHV,
    MedianBlur,
    GenInstanceUnetMap,
    GenInstanceContourMap,
    eqRGB2HED,
    eqHistCV,
    pipeHEDAugment,
    linearAugmentation,
)

from misc.info import COLOR_PALETE, MAP_TYPES, MODEL_TYPES, MODEL_PARAMS
from misc.utils import get_best_chkpts

####
class Config(object):
    def __init__(self, verbose=True):
        if os.path.exists('config.yml'):
            data_config = defaultdict(
                lambda: None,
                yaml.load(open("config.yml"), Loader=yaml.FullLoader),
            )
        else:
            print("No config.yml file was found. Please create one and rename as 'config.yml'")
            sys.exit()

        self.model_config = data_config["profile"]

        input_prefix = '/data/input/' if data_config["input_prefix"] is None else data_config["input_prefix"]
        output_prefix = '/data/output/' if data_config["output_prefix"] is None else data_config["output_prefix"]

        # init model params
        self.seed = data_config["seed"] if data_config["seed"] is not None else 10
        self.model_type = data_config["model_type"] if data_config["model_type"] is not None else MODEL_TYPES[self.model_config]
        self.nr_classes = 2  # Nuclei Pixels vs Background
        self.nuclei_type_dict = data_config["nuclei_types"] if data_config["nuclei_types"] is not None else MAP_TYPES[self.model_config]
        self.nr_types = len(self.nuclei_type_dict.values()) + 1  # plus background

        # config_file = importlib.import_module("opt.hover") ### ./opt/hover
        # config_dict = config_file.__getattribute__(self.model_type)
        
        model_params = MODEL_PARAMS[self.model_type]
        for variable, value in model_params.items():
            self.__setattr__(variable, value)
    
        # Load config yml file
        self.log_path = output_prefix  # log root path
        self.data_dir_root = os.path.join(
            input_prefix, data_config["data_dir"]
        )  # without modes

        self.extract_type = data_config["extract_type"] if data_config["extract_type"] is not None else 'mirror'
        self.data_modes = data_config["data_modes"]
        
        self.img_ext = (
            ".png" if data_config["img_ext"] is None else data_config["img_ext"]
        )

        for step in ["extract", "train", "infer", "process", "export"]:
            exec(
                f"self.out_{step}_root = os.path.join(output_prefix, '{step}')"
            )
        # self.out_extract_root = os.path.join(output_prefix, 'extract')

        self.img_dirs = {
            k: v
            for k, v in zip(
                self.data_modes,
                [
                    os.path.join(self.data_dir_root, mode, "Images")
                    for mode in self.data_modes
                ],
            )
        }
        self.labels_dirs = {
            k: v
            for k, v in zip(
                self.data_modes,
                [
                    os.path.join(self.data_dir_root, mode, "Labels")
                    for mode in self.data_modes
                ],
            )
        }
        win_code = f"{self.model_config}_{self.win_size[0]}x{self.win_size[1]}_{self.step_size[0]}x{self.step_size[1]}"
        self.out_extract = {
            k: v
            for k, v in zip(
                self.data_modes,
                [
                    os.path.join(self.out_extract_root, win_code, mode, "Annotations")
                    for mode in self.data_modes
                ],
            )
        }

        self.color_palete = COLOR_PALETE

        # self.model_name = f"{data_config['profile']}-{data_config['input_augs']}-{data_config['id']}"
        self.model_name = f"{data_config['name']}-{data_config['id']}"

        self.data_ext = (
            ".npy" if data_config["data_ext"] is None else data_config["data_ext"]
        )
        # list of directories containing validation patches

        # self.train_dir = data_config['train_dir']
        # self.valid_dir = data_config['valid_dir']
        include_extract = True if data_config["include_extract"] is None else data_config["include_extract"]
        if include_extract:
            self.train_dir = [
                os.path.join(self.out_extract_root, win_code, x)
                for x in data_config["train_dir"]
            ]
            self.valid_dir = [
                os.path.join(self.out_extract_root, win_code, x)
                for x in data_config["valid_dir"]
            ]
        else:
            self.train_dir = [
                os.path.join(self.data_dir_root, x) for x in data_config["train_dir"]
            ]
            self.valid_dir = [
                os.path.join(self.data_dir_root, x) for x in data_config["valid_dir"]
            ]

        # nr of processes for parallel processing input
        self.nr_procs_train = (
            8
            if data_config["nr_procs_train"] is None
            else data_config["nr_procs_train"]
        )
        self.nr_procs_valid = (
            4
            if data_config["nr_procs_valid"] is None
            else data_config["nr_procs_valid"]
        )

        # normalize RGB to 0-1 range
        self.input_norm = data_config["input_norm"] if data_config["input_norm"] is not None else True

        # self.save_dir = os.path.join(output_prefix, 'train', self.model_name)
        self.save_dir = os.path.join(self.out_train_root, self.model_name)

        #### Info for running inference
        self.inf_auto_find_chkpt = data_config["inf_auto_find_chkpt"]
        # path to checkpoints will be used for inference, replace accordingly

        if self.inf_auto_find_chkpt:
            self.inf_model_path = os.path.join(self.save_dir)
        else:
            self.inf_model_path = os.path.join(
                input_prefix, "models", data_config["inf_model"]
            )
        # self.save_dir + '/model-19640.index'

        # output will have channel ordering as [Nuclei Type][Nuclei Pixels][Additional]
        # where [Nuclei Type] will be used for getting the type of each instance
        # while [Nuclei Pixels][Additional] will be used for extracting instances

        # TODO: encode the file extension for each folder?
        # list of [[root_dir1, codeX, subdirA, subdirB], [root_dir2, codeY, subdirC, subdirD] etc.]
        # code is used together with 'inf_output_dir' to make output dir for each set
        self.inf_imgs_ext = (
            ".png"
            if data_config["inf_imgs_ext"] is None
            else data_config["inf_imgs_ext"]
        )

        # rootdir, outputdirname, subdir1, subdir2(opt) ...
        self.inf_data_list = [
            os.path.join(input_prefix, x)
            for x in data_config["inf_data_list"]
        ]

        model_used = (
            self.model_name
            if self.inf_auto_find_chkpt
            else os.path.basename(f"{data_config['inf_model'].split('.')[0]}")
        )

        self.inf_auto_metric = data_config["inf_auto_metric"] if data_config["inf_auto_metric"] is not None else 'valid_dice'
        self.inf_output_dir = os.path.join(
            self.out_infer_root,
            f"{model_used}.{''.join(data_config['inf_data_list']).replace('/', '_').rstrip('_')}.{self.inf_auto_metric}",
        )
        self.model_export_dir = os.path.join(self.out_export_root, self.model_name)
        self.remap_labels = data_config["remap_labels"] if data_config["remap_labels"] is not None else True
        
        
        self.outline = data_config["outline"]
        self.skip_types = data_config["skip_types"]
        # self.skip_types = (
        #     [self.nuclei_type_dict[x.strip()] for x in data_config["skip_types"]]
        #     if data_config["skip_types"] is not None
        #     else None
        # )
        self.process_maping = data_config["process_maping"]

        self.inf_auto_comparator = data_config["inf_auto_comparator"] if data_config["inf_auto_comparator"] is not None else '>'
        self.inf_batch_size = data_config["inf_batch_size"] if data_config["inf_batch_size"] is not None else 16

        # For inference during evalutaion mode i.e run by inferer.py
        self.eval_inf_input_tensor_names = ["images"]
        self.eval_inf_output_tensor_names = ["predmap-coded"]
        # For inference during training mode i.e run by trainer.py
        self.train_inf_output_tensor_names = ["predmap-coded", "truemap-coded"]

        assert data_config["input_augs"] != "" or data_config["input_augs"] is not None

        #### Policies
        policies = {
            "p_standard": [
                imgaug.RandomApplyAug(
                    imgaug.RandomChooseAug(
                        [GaussianBlur(), MedianBlur(), imgaug.GaussianNoise(),]
                    ),
                    0.5,
                ),
                imgaug.RandomOrderAug(
                    [
                        imgaug.Hue((-8, 8), rgb=True),
                        imgaug.Saturation(0.2, rgb=True),
                        imgaug.Brightness(26, clip=True),
                        imgaug.Contrast((0.75, 1.25), clip=True),
                    ]
                ),
                imgaug.ToUint8(),
            ],
            "p_neutral": [
                
            ],
            "p_hed_random": [
                imgaug.RandomApplyAug(
                    imgaug.RandomChooseAug(
                        [
                            GaussianBlur(),
                            MedianBlur(),
                            imgaug.GaussianNoise(),
                            #
                            imgaug.ColorSpace(cv2.COLOR_RGB2HSV),
                            imgaug.ColorSpace(cv2.COLOR_HSV2RGB),
                            #
                            eqRGB2HED(),
                        ]
                    ),
                    0.5,
                ),
                # standard color augmentation
                imgaug.RandomOrderAug(
                    [
                        imgaug.Hue((-8, 8), rgb=True),
                        imgaug.Saturation(0.2, rgb=True),
                        imgaug.Brightness(26, clip=True),
                        imgaug.Contrast((0.75, 1.25), clip=True),
                    ]
                ),
                imgaug.ToUint8(),
            ],
            "p_linear_1": [
                imgaug.RandomApplyAug(
                    imgaug.RandomChooseAug(
                        [GaussianBlur(), MedianBlur(), imgaug.GaussianNoise(),]
                    ),
                    0.5,
                ),
                linearAugmentation(),
                imgaug.ToUint8(),
            ],
            "p_linear_2": [
                imgaug.RandomApplyAug(
                    imgaug.RandomChooseAug(
                        [GaussianBlur(), MedianBlur(), imgaug.GaussianNoise(),]
                    ),
                    0.5,
                ),
                linearAugmentation(),
                imgaug.RandomOrderAug(
                    [
                        imgaug.Hue((-8, 8), rgb=True),
                        imgaug.Saturation(0.2, rgb=True),
                        imgaug.Brightness(26, clip=True),
                        imgaug.Contrast((0.8, 1.20), clip=True),  # 0.75, 1.25
                    ]
                ),
                imgaug.ToUint8(),
            ],
            "p_linear_3": [
                imgaug.RandomApplyAug(
                    imgaug.RandomChooseAug(
                        [GaussianBlur(), MedianBlur(), imgaug.GaussianNoise(),]
                    ),
                    0.5,
                ),
                imgaug.RandomChooseAug(
                    [
                        linearAugmentation(),
                        imgaug.RandomOrderAug(
                            [
                                imgaug.Hue((-2, 2), rgb=True),
                                imgaug.Saturation(0.2, rgb=True),
                                imgaug.Brightness(26, clip=True),
                                imgaug.Contrast((0.9, 1.1), clip=True),  # 0.75, 1.25
                            ]
                        ),
                    ]
                ),
                imgaug.ToUint8(),
            ],
        }

        self.input_augs = policies[(data_config["input_augs"])]

        # Checks
        if verbose:
            print("--------")
            print("Config info:")
            print("--------")
            print(f"Log path: <{self.log_path}>")
            print(f"Extraction out dirs: <{self.out_extract}>")
            print("--------")
            print("Training")
            print(f"Model name: <{self.model_name}>")
            print(f"Input img dirs: <{self.img_dirs}>")
            print(f"Input labels dirs: <{self.labels_dirs}>")
            print(f"Train out dir: <{self.save_dir}>")
            print("--------")
            print("Inference")
            print(f"Auto-find trained model: <{self.inf_auto_find_chkpt}>")
            print(f"Inference model path dir: <{self.inf_model_path}>")
            print(f"Input inference path: <{self.inf_data_list}>")
            print(f"Output inference path: <{self.inf_output_dir}>")
            print(f"Model export out: <{self.model_export_dir}>")
            print("--------")
            print()
        ####

    def get_model(self):
        if self.model_type == "np_hv":
            model_constructor = importlib.import_module("model.hover")
            model_constructor = model_constructor.Model_NP_HV
        elif self.model_type == "np_hv_opt":
            model_constructor = importlib.import_module("model.hover_opt")
            model_constructor = model_constructor.Model_NP_HV_OPT
        return model_constructor  # NOTE return alias, not object

    # refer to https://tensorpack.readthedocs.io/modules/dataflow.imgaug.html for
    # information on how to modify the augmentation parameters
    def get_train_augmentors(self, input_shape, output_shape, view=False):
        shape_augs = [
            imgaug.Affine(
                shear=5,  # in degree
                scale=(0.8, 1.2),
                rotate_max_deg=179,
                translate_frac=(0.01, 0.01),
                interp=cv2.INTER_NEAREST,
                border=cv2.BORDER_CONSTANT,
            ),
            imgaug.Flip(vert=True),
            imgaug.Flip(horiz=True),
            imgaug.CenterCrop(input_shape),
        ]

        input_augs = self.input_augs

        label_augs = []
        if self.model_type == "np_hv" or self.model_type == "np_hv_opt":
            label_augs = [GenInstanceHV(crop_shape=output_shape)]
        if self.model_type == "np_dist":
            label_augs = [GenInstanceDistance(crop_shape=output_shape, inst_norm=True)]

        if not view:
            label_augs.append(imgaug.CenterCrop(output_shape))

        return shape_augs, input_augs, label_augs

    def get_valid_augmentors(self, input_shape, output_shape, view=False):
        shape_augs = [
            imgaug.CenterCrop(input_shape),
        ]

        input_augs = None

        label_augs = []
        if self.model_type == "np_hv" or self.model_type == "np_hv_opt":
            label_augs = [GenInstanceHV(crop_shape=output_shape)]
        if self.model_type == "np_dist":
            label_augs = [GenInstanceDistance(crop_shape=output_shape, inst_norm=True)]
        label_augs.append(BinarizeLabel())

        if not view:
            label_augs.append(imgaug.CenterCrop(output_shape))

        return shape_augs, input_augs, label_augs

    def gen_pred_config(self):
        if self.inf_auto_find_chkpt:
            self.inf_model_path = os.path.join(
                self.save_dir,
                str(
                    max(
                        [
                            int(x)
                            for x in [
                                name
                                for name in os.listdir(self.save_dir)
                                if os.path.isdir(os.path.join(self.save_dir, name))
                            ]
                        ]
                    )
                ),
            )
            print(f"Inference model path: <{self.inf_model_path}>")
            print(
                '-----Auto Selecting Checkpoint Basing On "%s" Through "%s" Comparison'
                % (self.inf_auto_metric, self.inf_auto_comparator)
            )
            model_path, stat = get_best_chkpts(
                self.inf_model_path, self.inf_auto_metric, self.inf_auto_comparator
            )
            print("Selecting: %s" % model_path)
            print("Having Following Statistics:")
            for key, value in stat.items():
                print("\t%s: %s" % (key, value))
            sess = get_model_loader(model_path)
        else:
            model_path = self.inf_model_path
            sess = SaverRestoreRelaxed(self.inf_model_path)

        model_constructor = self.get_model()
        pred_config = PredictConfig(
            model=model_constructor(),
            session_init=sess,
            input_names=self.eval_inf_input_tensor_names,
            output_names=self.eval_inf_output_tensor_names,
        )
        return pred_config


if __name__ == "__main__":
    config = Config()
    print (config.infer_mask_shape)
    print (config.win_size)
    print (config.optimizer)