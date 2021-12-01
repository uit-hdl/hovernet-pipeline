MAP_TYPES = {
    "hv_consep": {
        "Inflammatory": 2,  # 1
        "Epithelial": 3,  # 2
        "Spindle": 4,  # 3
        "Misc": 1,  # 4
    },
    "hv_pannuke": {
        "Inflammatory": 1,  # 1
        "Epithelial": 4,  # 2
        "Neoplastic cells": 5,  # 3
        "Connective": 2,  # 4
        "Dead cells": 3,  # 5
    },
    "hv_monusac": {
        "Epithelial": 1, 
        "Lymphocyte": 2, 
        "Macrophage": 3, 
        "Neutrophil": 4,
    },
}

COLOR_PALETE = {
    "Inflammatory": (0.0, 255.0, 0.0),  # bright green
    "Dead cells": (255.0, 255.0, 0.0),  # bright yellow
    "Neoplastic cells": (255.0, 0.0, 0.0),  # red  # aka Epithelial malignant
    "Epithelial": (0.0, 0.0, 255.0),  # dark blue  # aka Epithelial healthy
    "Misc": (0.0, 0.0, 0.0),  # pure black  # aka 'garbage class'
    "Spindle": (0.0, 255.0, 255.0),  # cyan  # Fibroblast, Muscle and Endothelial cells
    "Connective": (0.0, 220.0, 220.0),  # darker cyan    # Connective plus Soft tissue cells
    "Background": (255.0, 0.0, 170.0),  # pink
    ###
    "Lymphocyte": (170.0, 255.0, 0.0),  # light green
    "Macrophage": (170.0, 0.0, 255.0),  # purple
    "Neutrophil": (255.0, 170.0, 0.0),  # orange
    "black": (32.0, 32.0, 32.0),  # black
}

# orignal size (win size) - input size - output size (step size)
# 540x540 - 270x270 - 80x80  hover
# 512x512 - 256x256 - 164x164 hover_opt
# DATA_CODE = {
#     'np_hv_opt': '512x512_164x164',
#     'np_hv'    : '540x540_80x80'
# }

MODEL_TYPES = {"hv_consep": "np_hv", "hv_pannuke": "np_hv_opt", "hv_monusac": "np_hv_opt"}

STEP_SIZE = {"hv_consep": (80,80), "hv_pannuke": (164,164), "hv_monusac": (164,164)}

WIN_SIZE = {"hv_consep": (540,540), "hv_pannuke": (512,512), "hv_monusac": (512,512)}

# INPUT_SHAPE = { # WIN/2
#     'hv_consep': 270,
#     'hv_pannuke': 256,
#     'hv_monusac': 256,

# }
# MASK_SHAPE =  { # = STEP_SIZE
#     'hv_consep': 80,
#     'hv_pannuke': 164
#     'hv_monusac': 164
# }
