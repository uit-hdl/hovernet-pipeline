import glob
import os
from ast import literal_eval
import argparse


def rename(d, keymap):
    new_dict = {}
    for key, value in zip(d.keys(), d.values()):
        new_key = keymap.get(key, key)
        new_dict[new_key] = d[key]
    return new_dict


def process_logs(path, map_list, output_file='/tmp/counts.txt', tp='all'):
    '''
    tp argument can be any element in map_list, 'all' will display all types counts
    '''
    for file in glob.glob(os.path.join(path,"*.log")):
        with open(file) as f:
            log = f.read()
            name = substitute_string_name(log.split(' : ')[0])

            d = literal_eval(log.split(' : ')[1])
            d = {str(k):int(v) for k,v in d.items()}
            d = rename(d, dict(zip(d.keys(), map_list)))
            with open(output_file, 'w') as wf:
                if (tp == 'all'):
                    print (f'{name} %-% {d}', file=wf)
                else:
                    print (f'{name}, {d[tp]}', file=wf)


def substitute_string_name(name):
    chunks = name.split('_')
    id_name = '_'.join([chunks[0], chunks[1], chunks[2]])
    id_patch = '_'.join([chunks[3], chunks[4]])
    coords = '_'.join([chunks[5], chunks[6], chunks[7], chunks[8]])
    h_info = '_'.join([chunks[9], chunks[10]])
    ext = chunks[-1].split('.')[0] + 'PNG'
    return ('.'.join([id_name, id_patch, coords, h_info, ext]))


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--out', default='/tmp/counts.txt', help='File to write result.')
    parser.add_argument('--map_list', required=True, choices=['consep','pannuke', 'monusac'])
    parser.add_argument('--tp', default='all')
    return parser.parse_args()


if __name__ == "__main__":
    conf = get_arguments()
    if conf.map_list == 'consep':
        map_list = ['Misc', 'Inflammatory', 'Epithelial', 'Spindle']
    elif conf.map_list == 'pannuke':
        map_list = ['Inflammatory', 'Connective', 'Dead cells', 'Epithelial', 'Neoplastic cells']
    if conf.map_list == 'monusac':
        map_list = ["Epithelial", "Lymphocyte", "Macrophage", "Neutrophil"]

    process_logs(conf.path, map_list, output_file=conf.out, tp=conf.tp)