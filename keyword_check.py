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

vocab = set()
with open('./data/vpkl_keywords.txt', 'r') as f:
    for keyword in f:
        vocab.add(keyword.strip())
print(len(vocab))

base = set()
with open('./data/34_keywords.txt', 'r') as f:
    for keyword in f:
        base.add(keyword.strip())
print(len(base))

print(len(vocab.intersection(base)))
print(base - vocab)

labels_to_images = np.load(Path('data/gold_labels_to_images.npz'), allow_pickle=True)['labels_to_images'].item()
print(len(labels_to_images))
print(labels_to_images.keys())

key = np.load(Path('data/gold_label_key.npz'), allow_pickle=True)['id_to_word_key'].item()
print(key)