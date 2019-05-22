# Copyright 2019 SAP SE
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import numpy as np
import torch
import copy
import torchvision.datasets as dset
import torchvision.transforms as transforms


def load_data(dataset_train, dataset_test, classes_test, classes_train):
    dataset_train_ = copy.deepcopy(dataset_train)
    dataset_test_ = copy.deepcopy(dataset_test)
    idx = np.nonzero(np.isin(dataset_train.labels, classes_train))[0]
    dataset_train_.labels = np.array(dataset_train.labels).squeeze()[idx]
    dataset_train_.data = dataset_train.data[idx]

    idx_test = np.nonzero(np.isin(dataset_test.labels, classes_test))[0]
    dataset_test_.labels = np.array(dataset_test.labels).squeeze()[idx_test]
    dataset_test_.data = dataset_test.data[idx_test]

    return dataset_train_, dataset_test_


def get(seed=0, data_root=None, fixed_order=False, pc_valid=0.1, n_classes=1, imageSize=None):
    ncla = n_classes
    size = imageSize
    data = {}
    if not os.path.isdir(data_root + '/split_svhn_' + str(size) + '/'):
        os.makedirs(data_root + '/split_svhn_' + str(size) + '/')
        dataset_train = dset.SVHN(root=data_root, split='train',download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5)),
                                ]))
        dataset_test = dset.SVHN(root=data_root, split='test',download=True,
                                    transform=transforms.Compose([
                                        #transforms.Resize(imageSize),
                                        #transforms.CenterCrop(imageSize),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5),
                                                             (0.5, 0.5, 0.5)),
                                    ]))
        for n in range(10):
            print("Loading data task: ", n)
            data[n] = {}
            data[n]['name'] = str(n)
            data[n]['ncla'] = ncla
            dataset_train_, dataset_test_ = load_data(dataset_train,dataset_test, [n], [n])
            train_length = int((1 - pc_valid) * len(dataset_train_.labels))
            valid_length = len(dataset_train_) - train_length
            dataset_train__, valid_subset = torch.utils.data.random_split(dataset_train_, (train_length, valid_length))
            loader_train = torch.utils.data.DataLoader(dataset_train__, batch_size=len(dataset_train__), shuffle=False)
            loader_test = torch.utils.data.DataLoader(dataset_test_, batch_size=len(dataset_test_), shuffle=False)
            loader_valid = torch.utils.data.DataLoader(valid_subset, batch_size=len(valid_subset), shuffle=False)
            x_train, y_train = next(iter(loader_train))
            x_test, y_test = next(iter(loader_test))
            x_valid, y_valid = next(iter(loader_valid))
            data[n]['train'] = {'x': x_train, 'y': y_train}
            data[n]['valid'] = {'x': x_valid, 'y': y_valid}
            data[n]['test'] = {'x': x_test, 'y': y_test}

            for s in ['train', 'test', 'valid']:
                #data[n][s]['x'] = torch.stack(data[n][s]['x'])
                data[n][s]['y'] = torch.LongTensor(np.array(data[n][s]['y'], dtype=int)).view(-1)
                print(data[n][s]['y'])
                # SAVE
                torch.save(data[n][s]['x'], os.path.join(os.path.expanduser(data_root + '/split_svhn_' + str(size)),
                                                         'data' + str(n) + s + 'x.bin'))
                torch.save(data[n][s]['y'], os.path.join(os.path.expanduser(data_root + '/split_svhn_' + str(size)),
                                                         'data' + str(n) + s + 'y.bin'))

    else:
        # Load binary files
        for idx in range(10):
            data[idx] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
            data[idx]['name'] = str(idx)
            data[idx]['ncla'] = ncla

            # Load
            for s in ['train', 'test', 'valid']:
                data[idx][s] = {'x': [], 'y': []}
                data[idx][s]['x'] = torch.load(
                    os.path.join(os.path.expanduser(data_root + '/split_svhn_' + str(size)),
                                 'data' + str(idx) + s + 'x.bin'))
                # data[idx][s]['x'] = torch.LongTensor(data[idx][s]['x'])
                data[idx][s]['y'] = torch.load(
                    os.path.join(os.path.expanduser(data_root + '/split_svhn_' + str(size)),
                                 'data' + str(idx) + s + 'y.bin'))
                # data[idx][s]['y'] = torch.LongTensor(data[idx][s]['y'])
    data_test = data
    return data, None, None