import cv2
import math
import random
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from .utils import bounding_box

####
def gen_colors(N, random_colors=True, bright=True):
    """
    Generate colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    if random_colors:
        random.shuffle(colors)
    return np.array(colors) * 255


# def gen_colors_outline(N, outline_idx=2):
#     """
#     Generate colors.
#     To get visually distinct colors, generate them in HSV space then
#     convert to RGB.
#     Outline specific color
#     """
#     pallete_bright_1 = np.array([[255.0, 0.0, 0.0], # red
#                                 [204.0, 255.0, 0.0], # greeny yellow
#                                 [0.0, 255.0, 102.0], # green - Inflammatory
#                                 [0.0, 102.0, 255.0], # blue - Epithelial
#                                 [204.0, 0.0, 255.0]]) # pink

#     pallete_bright_2 = np.array([[255.0, 0.0, 0.0], # bright red
#                                 [255.0, 255.0, 0.0], # bright yellow
#                                 [0.0, 255.0, 0.0], # bright green
#                                 [0.0, 255.0, 255.0], # bright cyan
#                                 [0.0, 0.0, 255.0], # bright blue
#                                 [255.0, 0.0, 255.0]]) # bright pink

#     compare_pallete = np.array([[255.0, 0.0, 0.0], # bright red
#                                 [255.0, 255.0, 0.0], # bright yellow Neoplastic
#                                 [0.0, 255.0, 0.0], # bright green Inflammatory
#                                 [0.0, 255.0, 255.0], # bright cyan Connective
#                                 [255.0, 0.0, 255.0], # bright pink Dead cells
#                                 [0.0, 0.0, 255.0]]) # bright blue Epithelial


#     brightness = 0.6
#     hsv = [(i / N, 1, 1.0) if i == outline_idx else (i / N, 1, brightness) for i in range(N)]
#     colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
#     # return colors
#     return list(compare_pallete / 255.0)

####
def visualize_instances(mask, canvas=None, color_info=None, to_outline=None, skip=None):
    """
    Args:
        mask: array of NW
        color_info: tuple ((cfg.nuclei_type_dict, cfg.color_palete), pred_inst_type[:, None])
    Return:
        Image with the instance overlaid
    """

    canvas = (
        np.full(mask.shape + (3,), 200, dtype=np.uint8)
        if canvas is None
        else np.copy(canvas)
    )

    insts_list = list(np.unique(mask))  # [0,1,2,3,4,..,820]
    insts_list.remove(0)  # remove background

    if color_info is None:
        inst_colors = gen_colors(len(insts_list))

    if color_info is not None:
        unique_colors = {}
        for key in color_info[0][0].keys():  # type_dict
            if (bool(to_outline) is True) and (key != to_outline):
                unique_colors[color_info[0][0][key]] = [224.0, 224.0, 224.0]
            else:
                unique_colors[color_info[0][0][key]] = color_info[0][1][
                    key
                ]  # color palete

    for idx, inst_id in enumerate(insts_list):
        if (color_info[1][idx][0]) == 0:  # if background inst
            continue

        if (skip is not None) and (
            (color_info[1][idx][0]) in skip
        ):  # if we skip specific type
            continue

        if color_info is None:
            inst_color = inst_colors[idx]
        else:
            inst_color = unique_colors[int(color_info[1][idx][0])]

        inst_map = np.array(mask == inst_id, np.uint8)
        y1, y2, x1, x2 = bounding_box(inst_map)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= mask.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= mask.shape[0] - 1 else y2
        inst_map_crop = inst_map[y1:y2, x1:x2]
        inst_canvas_crop = canvas[y1:y2, x1:x2]
        contours = cv2.findContours(
            inst_map_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # For opencv-python >= 4.1.0.25
        cv2.drawContours(inst_canvas_crop, contours[0], -1, inst_color, 2)

        # cv2.drawContours(inst_canvas_crop, contours[1], -1, inst_color, 2)
        canvas[y1:y2, x1:x2] = inst_canvas_crop
    return canvas


####
def gen_figure(
    imgs_list,
    titles,
    fig_inch,
    shape=None,
    share_ax="all",
    show=False,
    colormap=plt.get_cmap("jet"),
):

    num_img = len(imgs_list)
    if shape is None:
        ncols = math.ceil(math.sqrt(num_img))
        nrows = math.ceil(num_img / ncols)
    else:
        nrows, ncols = shape

    # generate figure
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=share_ax, sharey=share_ax)
    axes = [axes] if nrows == 1 else axes

    # not very elegant
    idx = 0
    for ax in axes:
        for cell in ax:
            cell.set_title(titles[idx])
            cell.imshow(imgs_list[idx], cmap=colormap)
            cell.tick_params(
                axis="both",
                which="both",
                bottom="off",
                top="off",
                labelbottom="off",
                right="off",
                left="off",
                labelleft="off",
            )
            idx += 1
            if idx == len(titles):
                break
        if idx == len(titles):
            break

    fig.tight_layout()
    return fig


####
