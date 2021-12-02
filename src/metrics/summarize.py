import glob
import argparse

class Patient:
    def __init__(self, filename, count):
        self.num = None
        self.images = {}
        self.add_file(filename, count)

    def add_file(self, filename, count):
        split_fname = filename.split("_")

        area = float(split_fname[3][1:])

        self.images[filename] = int(count) / (area / (1000*1000))
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


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="File from counts.py script", required=True)
    return parser.parse_args()

if __name__ == "__main__":
    conf = get_arguments()
    patients = load_patients(conf.f)
    for p in sorted(patients.keys()):
        p_real = patients[p]
        minimum = str(p_real.get_min_count())
        maximum = str(p_real.get_max_count())
        median = str(p_real.get_median_count())
        average = str(p_real.get_average_count())
        output = ",".join([p_real.num, minimum, maximum, median, average])
        print(output)