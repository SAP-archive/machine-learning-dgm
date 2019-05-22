#Copyright 2019 SAP SE
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
import os,sys
import os.path
import numpy as np
import random
import torch
import torch.utils.data
from torchvision import datasets, transforms
from sklearn.utils import shuffle
import urllib.request
from PIL import Image
import pickle
import utils

from lib import data_manager
from lib import data_io

import numpy as np
import keras
from keras.utils import np_utils

from keras.datasets import mnist


def split_dataset_by_labels(X, y, task_labels, nb_classes=None, multihead=False):
    """Split dataset by labels.

    Args:
        X: data
        y: labels
        task_labels: list of list of labels, one for each dataset
        nb_classes: number of classes (used to convert to one-hot)
    Returns:
        List of (X, y) tuples representing each dataset
    """
    if nb_classes is None:
        nb_classes = len(np.unique(y))
    datasets = []
    for labels in task_labels:
        idx = np.in1d(y, labels)
        if multihead:
            label_map = np.arange(nb_classes)
            label_map[labels] = np.arange(len(labels))
            data = X[idx], np_utils.to_categorical(label_map[y[idx]], len(labels))
        else:
            data = X[idx], np_utils.to_categorical(y[idx], nb_classes)
        datasets.append(data)
    return datasets


def construct_split_mnist(task_labels,  split='train', multihead=False):
    """Split MNIST dataset by labels.

        Args:
                task_labels: list of list of labels, one for each dataset
                split: whether to use train or testing data

        Returns:
            List of (X, y) tuples representing each dataset
    """
    # Load MNIST data and normalize
    nb_classes = 10
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28)
    X_test = X_test.reshape(-1, 28, 28)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    if split == 'train':
        X, y = X_train, y_train
    else:
        X, y = X_test, y_test

    return split_dataset_by_labels(X, y, task_labels, nb_classes, multihead)



def get(seed=0, data_root=None, fixed_order=False, pc_valid=0.15, n_classes=1, imageSize=None):
    print("Getting")
    binary = False
    if n_classes == 1:
        binary = True
        ncla=1
        task_labels = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]

    elif n_classes == 2:
        ncla=2
        task_labels = [[0,1], [2,3], [4,5], [6,7], [8,9] ]

    else:
        ncla = 10
        task_labels = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

    data={}
    taskcla=[]
    size=imageSize # -size of the maximum input, smaller once will be padded with 0
    n_tasks = len(task_labels)  
    training_datasets = construct_split_mnist(task_labels, split='train', multihead=binary)
    validation_datasets = construct_split_mnist(task_labels, split='test', multihead  = binary)

    if not os.path.isdir(data_root+'/'):
        os.makedirs(data_root+'/')
        for n, idx in enumerate(range(n_tasks)):
            dat = {}
            dat['train'] = Split_MNIST_loader( training_datasets[idx], idx, transform=transforms.Compose([transforms.ToPILImage(), transforms.Resize(imageSize),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ]), binary=binary)
            dat['test'] = Split_MNIST_loader( validation_datasets[idx], idx, transform=transforms.Compose([transforms.ToPILImage(), transforms.Resize(imageSize),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ]), binary= binary)
            data[n] = {}
            data[n]['name'] = str(idx)
            data[n]['ncla'] = ncla

            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[n][s] = {'x': [], 'y': []}

                for image, target in loader:
                    data[n][s]['x'].append(image[0])
                    data[n][s]['y'].append(target.numpy()[0])

            for s in ['train', 'test']:
                    # Expand to 5000
                    data[n][s]['x'] = torch.stack(data[n][s]['x'])
                    data[n][s]['y'] = torch.LongTensor(np.array(data[n][s]['y'], dtype=int)).view(-1)
                    #SAVE
                    torch.save(data[n][s]['x'], os.path.join(os.path.expanduser(data_root), 'data' + str(idx) + s + 'x.bin'))
                    torch.save(data[n][s]['y'], os.path.join(os.path.expanduser(data_root), 'data' + str(idx) + s + 'y.bin'))
    else:
        # Load binary files
        for n, idx in enumerate(range(n_tasks)):
            data[n] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
            data[n]['name'] = str(idx)
            data[n]['ncla'] = ncla

            # Load
            for s in ['train', 'test']:
                data[n][s] = {'x': [], 'y': []}
                data[n][s]['x'] = torch.load(
                    os.path.join(os.path.expanduser(data_root), 'data' + str(idx) + s + 'x.bin'))
                data[n][s]['y'] = torch.load(
                    os.path.join(os.path.expanduser(data_root), 'data' + str(idx) + s + 'y.bin'))

    # Validation
    for t in data.keys():
        r = np.arange(data[t]['train']['x'].size(0))
        print("data[t]['train']['x'].size(0)", data[t]['train']['x'].shape)
        print(r)
        r = np.array(shuffle(r, random_state=seed), dtype=int)
        nvalid = int(pc_valid * len(r))
        print(nvalid)

        ivalid = torch.LongTensor(r[:nvalid])
        itrain = torch.LongTensor(r[nvalid:])

        data[t]['valid'] = {}
        data[t]['valid']['x'] = data[t]['train']['x'][ivalid].clone()
        data[t]['valid']['y'] = data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x'] = data[t]['train']['x'][itrain].clone()
        data[t]['train']['y'] = data[t]['train']['y'][itrain].clone()

    # Others
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    #data['ncla'] = n

    return data, taskcla, size

class Split_MNIST_loader(torch.utils.data.Dataset):

    def __init__(self, data, task, train=True, transform=None, max_samples=float('Inf'), seed=0, binary=True):
        self.transform = transform
        random.seed(seed)
        self.data = data[0]
        self.labels = data[1]
        if not binary:
            if len(self.labels.shape) > 1:
                self.labels = np.where(self.labels == 1)[1] #- 2*task

    def __getitem__(self, index):
        """
        Args: index (int): Index
        Returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = img.reshape(1, 28, 28)
        img = torch.Tensor(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)