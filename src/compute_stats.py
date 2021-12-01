import argparse
import cProfile as profile
import glob
import os

import cv2
import numpy as np
import scipy.io as sio
import pandas as pd

from misc.info import MAP_TYPES

from metrics.stats_utils import *


def run_nuclei_type_stat(
    pred_dir, true_dir, nuclei_type_dict, type_uid_list=None, exhaustive=True, rad=12
):
    """
    rad = 12 if x40
    rad = 6 if x20
    """

    file_list = glob.glob(os.path.join(pred_dir, "*.mat"))
    file_list.sort()  # ensure same order [1]
    paired_all = []  # unique matched index pair
    unpaired_true_all = []  # the index must exist in `true_inst_type_all` and unique
    unpaired_pred_all = []  # the index must exist in `pred_inst_type_all` and unique
    true_inst_type_all = []  # each index is 1 independent data point
    pred_inst_type_all = []  # each index is 1 independent data point

    for file_idx, filename in enumerate(file_list[:]):
        filename = os.path.basename(filename)
        basename = filename.split(".")[0]
        # print (basename)
        # true_info = sio.loadmat(os.path.join(true_dir, '{}.mat'.format(basename)))
        # # dont squeeze, may be 1 instance exist
        # true_centroid  = (true_info['inst_centroid']).astype('float32')
        # true_inst_type = (true_info['inst_type']).astype('int32')

        true_info = np.load(
            os.path.join(true_dir, "{}.npy".format(basename)), allow_pickle=True
        )
        # dont squeeze, may be 1 instance exist
        true_centroid = (true_info.item().get("inst_centroid")).astype("float32")
        true_inst_type = (true_info.item().get("inst_type")).astype("int32")
        if true_centroid.shape[0] != 0:
            true_inst_type = true_inst_type[:, 0]
        else:  # no instance at all
            pass
            true_centroid = np.array([[0, 0]])
            true_inst_type = np.array([0])

        pred_info = sio.loadmat(os.path.join(pred_dir, "{}.mat".format(basename)))
        # dont squeeze, may be 1 instance exist
        pred_centroid = (pred_info["inst_centroid"]).astype("float32")
        pred_inst_type = (pred_info["inst_type"]).astype("int32")

        if pred_centroid.shape[0] != 0:
            pred_inst_type = pred_inst_type[:, 0]
        else:  # no instance at all
            pass
            pred_centroid = np.array([[0, 0]])
            pred_inst_type = np.array([0])

        # ! if take longer than 1min for 1000 vs 1000 pairing, sthg is wrong with coord
        paired, unpaired_true, unpaired_pred = pair_coordinates(
            true_centroid, pred_centroid, rad
        )

        # * Aggreate information
        # get the offset as each index represent 1 independent instance
        true_idx_offset = (
            true_idx_offset + true_inst_type_all[-1].shape[0] if file_idx != 0 else 0
        )
        pred_idx_offset = (
            pred_idx_offset + pred_inst_type_all[-1].shape[0] if file_idx != 0 else 0
        )
        true_inst_type_all.append(true_inst_type)
        pred_inst_type_all.append(pred_inst_type)

        # increment the pairing index statistic
        if paired.shape[0] != 0:  # ! sanity
            paired[:, 0] += true_idx_offset
            paired[:, 1] += pred_idx_offset
            paired_all.append(paired)

        unpaired_true += true_idx_offset
        unpaired_pred += pred_idx_offset
        unpaired_true_all.append(unpaired_true)
        unpaired_pred_all.append(unpaired_pred)

    paired_all = np.concatenate(paired_all, axis=0)
    unpaired_true_all = np.concatenate(unpaired_true_all, axis=0)
    unpaired_pred_all = np.concatenate(unpaired_pred_all, axis=0)
    true_inst_type_all = np.concatenate(true_inst_type_all, axis=0)
    pred_inst_type_all = np.concatenate(pred_inst_type_all, axis=0)

    paired_true_type = true_inst_type_all[paired_all[:, 0]]
    paired_pred_type = pred_inst_type_all[paired_all[:, 1]]
    unpaired_true_type = true_inst_type_all[unpaired_true_all]
    unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]

    ###
    def _f1_type(paired_true, paired_pred, unpaired_true, unpaired_pred, type_id, w):
        type_samples = (paired_true == type_id) | (paired_pred == type_id)

        paired_true = paired_true[type_samples]
        paired_pred = paired_pred[type_samples]

        tp_dt = ((paired_true == type_id) & (paired_pred == type_id)).sum()
        tn_dt = ((paired_true != type_id) & (paired_pred != type_id)).sum()
        fp_dt = ((paired_true != type_id) & (paired_pred == type_id)).sum()
        fn_dt = ((paired_true == type_id) & (paired_pred != type_id)).sum()

        if not exhaustive:
            ignore = (paired_true == -1).sum()
            fp_dt -= ignore

        fp_d = (unpaired_pred == type_id).sum()
        fn_d = (unpaired_true == type_id).sum()

        # print (f"tp_dt: {tp_dt}") # TPc
        # print (f"tn_dt: {tn_dt}") # TNc
        # print (f"fp_dt: {fp_dt}") # FPc
        # print (f"fn_dt: {fn_dt}") # FNc
        # print (f"fp_d: {fp_d}")
        # print (f"fn_d: {fn_d}")

        f1_type = (2 * (tp_dt + tn_dt)) / (
            2 * (tp_dt + tn_dt)
            + w[0] * fp_dt
            + w[1] * fn_dt
            + w[2] * fp_d
            + w[3] * fn_d
        )

        # Only classification
        # precision_type =  tp_dt / (tp_dt + fp_dt)
        # recall_type = tp_dt / (tp_dt + fn_dt)
        # new_f1 = 2 * (precision_type * recall_type) / (precision_type + recall_type)

        tp_w = tp_dt + tn_dt
        fp_w = 2 * fp_dt + fp_d
        fn_w = 2 * fn_dt + fn_d

        # precision_type = tp_w / (tp_w + fn_w)
        # recall_type = tp_w / (tp_w + fp_w)
        # new_f1 = 2 * (precision_type * recall_type) / (precision_type + recall_type)

        precision_type = tp_w / (tp_w + fp_w)
        recall_type = tp_w / (tp_w + fn_w)
        new_f1 = (
            2 * (precision_type * recall_type) / (precision_type + recall_type)
        )  # just check, same as f1_type

        return f1_type, precision_type, recall_type, new_f1

    # overall
    # * quite meaningless for not exhaustive annotated dataset
    w = [1, 1]
    tp_d = paired_pred_type.shape[0]
    fp_d = unpaired_pred_type.shape[0]
    fn_d = unpaired_true_type.shape[0]

    tp_tn_dt = (paired_pred_type == paired_true_type).sum()
    fp_fn_dt = (paired_pred_type != paired_true_type).sum()

    if not exhaustive:
        ignore = (paired_true_type == -1).sum()
        fp_fn_dt -= ignore

    acc_type = tp_tn_dt / (tp_tn_dt + fp_fn_dt)

    # precision/recall
    precision = tp_d / (tp_d + w[0] * fp_d)
    recall = tp_d / (tp_d + w[0] * fn_d)

    f1_d = 2 * tp_d / (2 * tp_d + w[0] * fp_d + w[1] * fn_d)

    w = [2, 2, 1, 1]

    if type_uid_list is None:
        type_uid_list = np.unique(true_inst_type_all).tolist()
        if 0 in type_uid_list:
            type_uid_list.remove(0)
    print(f"true type_uid_list: {type_uid_list}")
    print(f"pred type_uid_list: {np.unique(pred_inst_type_all).tolist()}")
    results_list = [f1_d, acc_type, precision, recall]

    for type_uid in type_uid_list:
        f1_type, precision_type, recall_type, new_f1 = _f1_type(
            paired_true_type,
            paired_pred_type,
            unpaired_true_type,
            unpaired_pred_type,
            type_uid,
            w,
        )
        results_list.append(f1_type)

        print(f"{type_uid}_precision: {precision_type}")
        print(f"{type_uid}_recall: {recall_type}")
        print(f"{type_uid}_f1: {new_f1}")

    np.set_printoptions(formatter={"float": "{: 0.5f}".format})

    types = sorted([f"{v}_{k}" for k, v in nuclei_type_dict.items()])
    print(types)
    for k, v in zip(
        ("f1_d", "accuracy_type", "precision", "recall", *types), np.array(results_list)
    ):
        print(f"{k}: {v}")
    print()
    return


def run_nuclei_inst_stat(pred_dir, true_dir, print_img_stats=False):

    # print stats of each image
    file_list = glob.glob(os.path.join(pred_dir, "*.mat"))
    file_list.sort()  # ensure same order

    metrics = [[], [], [], [], [], []]
    for filename in file_list[:]:
        filename = os.path.basename(filename)
        basename = filename.split(".")[0]

        # true = sio.loadmat(os.path.join(true_dir, '{}.mat'.format(basename)))
        # true = (true['inst_map']).astype('int32')
        true = np.load(
            os.path.join(true_dir, "{}.npy".format(basename)), allow_pickle=True
        )
        true = (true.item().get("inst_map")).astype("int32")

        pred = sio.loadmat(os.path.join(pred_dir, "{}.mat".format(basename)))
        pred = (pred["inst_map"]).astype("int32")

        # to ensure that the instance numbering is contiguous
        pred = remap_label(pred, by_size=False)
        true = remap_label(true, by_size=False)

        # print (basename)
        pq_info = get_fast_pq(true, pred, match_iou=0.5)[0]
        metrics[0].append(get_dice_1(true, pred))
        metrics[1].append(get_fast_aji(true, pred))
        metrics[2].append(pq_info[0])  # dq
        metrics[3].append(pq_info[1])  # sq
        metrics[4].append(pq_info[2])  # pq
        metrics[5].append(get_fast_aji_plus(true, pred))

        if print_img_stats:
            print(basename, end="\t")
            for scores in metrics:
                print("%f " % scores[-1], end="  ")
            print()
    ####
    metrics = np.array(metrics)
    metrics_avg = np.mean(metrics, axis=-1)
    np.set_printoptions(formatter={"float": "{: 0.5f}".format})
    for entry in list(
        zip(["dice", "fast_aji", "dq", "pq", "sq", "aji_plus"], metrics_avg)
    ):
        print(f"{entry[0]}: {entry[1]}")
    metrics_avg = list(metrics_avg)
    return metrics


if __name__ == "__main__":
    # cfg = Config(verbose=False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", help="Point to output dir", required=True)
    parser.add_argument("--true_dir", help="Point to ground truth dir", required=True)
    parser.add_argument(
        "--map",
        help="Name of the nuclei type mapping",
        choices=["consep", "pannuke", "monusac"],
        required=True,
    )
    parser.add_argument(
        "--type", help="Run type stats", default=False, action="store_true"
    )
    parser.add_argument(
        "--inst", help="Run inst stats", default=False, action="store_true"
    )

    args = parser.parse_args()
    if args.type:
        print("---Type statistics---")
        run_nuclei_type_stat(
            args.pred_dir, args.true_dir, MAP_TYPES[f"hv_{args.map}"]
        )
    if args.inst:
        print("---Instance statistics---")
        run_nuclei_inst_stat(args.pred_dir, args.true_dir, print_img_stats=False)
