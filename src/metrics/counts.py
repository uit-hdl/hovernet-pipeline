import scipy.io as sio
import glob
import os
from ast import literal_eval
import colorsys
import numpy as np
import sys
import argparse


################################################################
# This script was used to get counts per patient for UiT_Dataset
################################################################

map_list = ["color_red", "lymphoid", "color_green", "cancercells", "fibroblasts"]


class Patient:
    def __init__(self, filename, count):
        self.num = None
        self.images = {}
        self.add_file(filename, count)

    def add_file(self, filename, count):
        split_fname = filename.split("_")

        area = float(split_fname[3][1:])

        self.images[filename] = int(count) / (area / (1000 * 1000))
        self.num = filename.split("_")[0]

    def get_average_count(self):
        return sum(self.images.values()) / len(self.images.values())

    def get_median_count(self):
        sorted_values = sorted(self.images.values())
        index = int(len(sorted_values) / 2)
        if len(sorted_values) % 2 == 0:
            return (sorted_values[index] + sorted_values[index - 1]) / 2
        else:
            return sorted_values[index]

    def get_min_count(self):
        return min(self.images.values())

    def get_max_count(self):
        return max(self.images.values())


def rename(d, keymap):
    new_dict = {}
    for key, value in zip(d.keys(), d.values()):
        new_key = keymap.get(key, key)
        new_dict[new_key] = d[key]
    return new_dict


def process_logs(tp="all"):  # lymphoid

    for file in glob.glob("*.log"):
        with open(file) as f:
            log = f.read()
            name = substitute_string_name(log.split(" : ")[0])

            d = literal_eval(log.split(" : ")[1])
            d = {str(k): int(v) for k, v in d.items()}
            d = rename(d, dict(zip(d.keys(), map_list)))
            if tp == "all":
                print(f"{name} %-% {d}")
            else:
                print(f"{name}, {d[tp]}")


def substitute_string_name(name):
    chunks = name.split("_")

    id_name = "_".join([chunks[0], chunks[1], chunks[2]])
    id_patch = "_".join([chunks[3], chunks[4]])
    coords = "_".join([chunks[5], chunks[6], chunks[7], chunks[8]])
    h_info = "_".join([chunks[9], chunks[10]])
    ext = chunks[-1].split(".")[0] + "PNG"

    return ".".join([id_name, id_patch, coords, h_info, ext])


def load_patients(filename):
    patients = {}
    with open(filename) as f:
        for line in f:
            target_file, count = line.split(",")
            patient = int(line.split("_")[0])
            if not patient in patients:
                patients[patient] = Patient(target_file, count)
            else:
                patients[patient].add_file(target_file, count)
    return patients


def load_patients_single_files(file_glob):
    patients = {}
    for f in glob.glob(file_glob):
        with open(f) as txtf:
            line = txtf.readline().strip()
            _, count = line.split(":")
            patient = int(f.split("_")[0])
            if not patient in patients:
                patients[patient] = Patient(f, count)
            else:
                patients[patient].add_file(f, count)
    return patients


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", help="Use single files format", action="store_true")
    parser.add_argument("-n", help="Use multiple files format", action="store_true")
    parser.add_argument("-f", help="Filename or glob", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    # process logs
    process_logs(tp="lymphoid")
    # print (np.array(gen_colors(5, random=False))*255)

    # counts per patient
    conf = get_arguments()
    if conf.e:
        patients = load_patients_single_files(conf.f)
    elif conf.n:
        patients = load_patients(conf.f)
    for p in sorted(patients.keys()):
        p_real = patients[p]
        minimum = str(p_real.get_min_count())
        maximum = str(p_real.get_max_count())
        median = str(p_real.get_median_count())
        average = str(p_real.get_average_count())
        output = ",".join([p_real.num, minimum, maximum, median, average])
        print(output)
