#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import argparse
import torch
from models.setup import *
from models.GeneralModels import *
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import scipy
import scipy.signal
import librosa
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def modelSetup(parser, test=False):

    config_file = parser.pop("config_file")
    print(f'configs/{config_library[config_file]}')
    with open(f'configs/{config_library[config_file]}') as file:
        args = json.load(file)

    image_base = parser.pop("image_base")

    for key in parser:
        args[key] = parser[key]

    args["data_train"] = Path(args["data_train"])
    args["data_val"] = Path(args["data_val"])
    args["data_test"] = Path(args["data_test"])

    getDevice(args)

    return args, image_base

command_line_args = {
    "resume": False, 
    "config_file": 'multilingual+matchmap',
    "device": "0", 
    "restore_epoch": -1, 
    "image_base": ".."
}

args, image_base = modelSetup(command_line_args)

image_labels = np.load(Path('data/gold_image_to_labels.npz'), allow_pickle=True)['image_labels'].item()
labels_to_images = np.load(Path('data/gold_labels_to_images.npz'), allow_pickle=True)['labels_to_images'].item()
key = np.load(Path('data/gold_label_key.npz'), allow_pickle=True)['id_to_word_key'].item()

with open(args["data_train"], 'r') as fp:
    data = json.load(fp)
image_base_path = Path(image_base).absolute()

id_lookup = {}
ids_used = set()
for fn in data:
    data_point = np.load(fn + ".npz")
    name = '_'.join(str(Path(fn).stem).split('_')[0:2])
    ids = np.unique(image_labels[fn.split('/')[-1].split('+')[0]])
    ids = list(ids)
    for id in ids:
        if id not in ids_used:
            ids_used.add(id)

for id in sorted(list(ids_used)):
    print(f'{id}: {key[id]}')

print(f'Number of keywords: {len(ids_used)}')