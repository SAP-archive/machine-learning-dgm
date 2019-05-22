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
from __future__ import print_function
from networks import net_DGMw_imnet as model
from approaches import DGMw_imnet as approach
import os
import random
import argparse
import shutil
import time
import datetime
import importlib
import numpy as np
from cfg.load_config import opt, cfg_from_file
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from utils.folder import ImageFolder

ts = time.time()

# Arguments
parser = argparse.ArgumentParser(description='xxx')
parser.add_argument(
    '--dataset',
    default='imnet',
    type=str,
    required=False,
    choices=['imagenet'],
    help='Dataset name')
parser.add_argument(
    '--cfg_file',
    default=None,
    type=str,
    required=False,
    help='Path to the configuration file')
cfg = parser.parse_args()
if cfg.cfg_file is not None:
    try:
        cfg_from_file(cfg.cfg_file)
    except FileNotFoundError:
        if cfg.dataset == "imnet":
            cfg_file = 'cfg/cfg_imnet_dgmw.yml'
            cfg_from_file(cfg_file)
else:
    if cfg.dataset == "imnet":
        cfg_file = 'cfg/cfg_imnet_dgmw.yml'
        cfg_from_file(cfg_file)

print(opt)
try:
    os.makedirs(opt.outf)
except OSError:
    pass

try:
    os.makedirs(opt.outf_models)
except OSError:
    pass


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
cuda1 = torch.device(opt.device_D)
cuda2 = torch.device(opt.device_G)

ngpu = int(1)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
num_classes = int(100)
nc = 3

# resnet18
netD = model.netD(2048, n_classes=10, device=cuda1)
netG = model.netG(nz, ngf, nc, opt.smax_g, device=cuda2)
print(netD)
print(netG)
ts = time.time()
log_dir = opt.log_dir + \
    datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
importlib.reload(approach)
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.makedirs(log_dir)

#idx = [1,15,29,45,59,65,81,89,90,99]
idx = opt.class_idx_imnet
appr = approach.App(
    model,
    netG,
    netD,
    log_dir,
    opt.outf,
    niter=opt.niter,
    batchSize=opt.batchSize,
    imageSize=opt.imageSize,
    nz=int(
        opt.nz),
    nb_label=num_classes,
    cuda=True,
    beta1=opt.beta1,
    lr_D=opt.lr_D,
    lr_G=opt.lr_G,
    lamb_G=opt.lamb_G,
    reinit_D=opt.reinit_D,
    lambd_adv=opt.lambda_adv,
    lambda_wassersten=opt.lambda_wasserstein,
    dataroot_test=opt.dataroot_val,
    dataroot=opt.dataroot,
    store_model=opt.store_models,
    out_models=opt.outf_models,
    calc_fid_imnet=opt.calc_fid_imnet,
    class_idx=idx)  # , gpu_tracker=gpu_tracker)
appr.writer.text_summary("opt", str(opt))


test_acc_tasks = []
conf_matrixes = []
for t in range(10):
    idx_ = [i + (t * 100) for i in idx]
    dataset = ImageFolder(
        root=opt.dataroot,
        transform=transforms.Compose([
            transforms.Resize(opt.imageSize),
            transforms.CenterCrop(opt.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        classes_idx=(idx_)
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(
            opt.workers))
    test_acc_task, conf_matrixes_task, mask_G = appr.train(
        dataloader, dataset, t, smax_g=opt.smax_g, use_aux_G=opt.aux_G)
    test_acc_tasks.append(test_acc_task)
    conf_matrixes.append(conf_matrixes_task)
