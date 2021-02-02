import os
import cv2
import glob
import staintools

from config import Config

from misc.utils import rm_n_mkdir


def stain_normilize(img_dir, save_dir, stain_norm_target, norm_brightness=False):
    file_list = glob.glob(os.path.join(img_dir, "*.png"))
    file_list.sort()

    if norm_brightness:
        standardizer = staintools.LuminosityStandardizer()
    stain_normalizer = staintools.StainNormalizer(method="vahadane")

    # dict of paths to target image and dir code to make output folder
    # {'/data/TCGA-21-5784-01Z-00-DX1.tif' : '5784'}
    # stain_norm_targets = {k : v for k, v in zip(glob.glob(os.path.join(targets_dir, '*.*')), range(len(glob.glob(os.path.join(targets_dir, '*.*')))))}
    # stain_norm_target = {target : '1'}

    target_img = cv2.imread(stain_norm_target)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    if norm_brightness:
        target_img = standardizer.standardize(target_img)
    stain_normalizer.fit(target_img)

    norm_dir = save_dir
    rm_n_mkdir(norm_dir)

    for img_path in file_list:
        filename = os.path.basename(img_path)
        basename = filename.split(".")[0]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if norm_brightness:
            img = standardizer.standardize(img)
        img = stain_normalizer.transform(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(norm_dir, "{}.png".format(basename)), img)
        print(f"Saved {os.path.join(norm_dir, '{}.png'.format(basename))}.")


if __name__ == "__main__":
    cfg = Config()

    if cfg.norm_brightness:
        assert cfg.norm_target is not None
        assert os.path.exists(cfg.norm_target)
        print(f"Normalization, using target: {cfg.norm_target}")
        for mode in cfg.img_dirs.keys():
            stain_normilize(
                cfg.img_dirs[mode],
                cfg.out_preproc[mode],
                cfg.norm_target,
                norm_brightness=cfg.norm_brightness,
            )
