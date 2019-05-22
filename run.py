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
from __future__ import print_function
import time,datetime,argparse,os,random
import shutil
import torch.utils.data
from cfg.load_config import opt, cfg_from_file
import numpy as np

ts = time.time()

# Arguments
parser=argparse.ArgumentParser(description='xxx')
parser.add_argument('--dataset',default='mnist',type=str,required=False, choices=['mnist','svhn'],help='Dataset name')
parser.add_argument('--method',default='DGMw',type=str,required=False, choices=['DGMa','DGMw'], help='Method to run.')
#parser.add_argument('--cfg_file',default=None,type=str,required=False, help='Path to the configuration file')
cfg=parser.parse_args()
if cfg.method =="DGMw":
    if cfg.dataset == "mnist":
        cfg_file = 'cfg/cfg_mnist_dgmw.yml'
        cfg_from_file(cfg_file)
    elif cfg.dataset == "svhn":
        cfg_file = 'cfg/cfg_svhn_dgmw.yml'
        cfg_from_file(cfg_file)
elif cfg.method =="DGMa":
    if cfg.dataset == "mnist":
        cfg_file = 'cfg/cfg_mnist_dgma.yml'
        cfg_from_file(cfg_file)
    elif cfg.dataset == "svhn":
        cfg_file = 'cfg/cfg_svhn_dgma.yml'
        cfg_from_file(cfg_file)
print(opt)

#######################################################################################################################
opt.device = torch.device("cuda:" + str(opt.device) if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(opt.device)
print(opt)


try:
    os.makedirs(opt.outf)
except OSError:
    pass
try:
    os.makedirs(opt.outf_models)
except OSError:
    pass
try:
    os.makedirs(opt.outf + '/mask_histo')
except:
    pass


if opt.dataset=="mnist":
    from dataloaders import split_MNIST as dataloader
elif opt.dataset=="svhn":
    from dataloaders import split_SVHN as dataloader
if opt.method == "DGMw":
    from networks import net_DGMw as model
    from approaches import DGMw as approach
elif opt.method == "DGMa":
    from networks import net_DGMa as model
    from approaches import DGMa as approach



if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(opt.manualSeed)


print('Load data...')
data, taskcla, inputsize = dataloader.get(seed=opt.manualSeed,data_root=opt.dataroot+str(opt.imageSize), n_classes=1, imageSize=opt.imageSize)
print('Input size =', inputsize, '\nTask info =', taskcla)
for t in range(10):
    data[t]['train']['y'].data.fill_(t)
    data[t]['test']['y'].data.fill_(t)
    data[t]['valid']['y'].data.fill_(t)

nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nb_label = 10
if opt.dataset == 'mnist':
    nc = 1
elif opt.dataset == 'svhn':
    nc = 3

#classes are added one by one, we innitialize G with one head output
netG = model.netG(nz, ngf, nc, opt.smax_g, n_classes=1)
print(netG)
netD = model.netD(ndf, nc)
print(netD)


log_dir = opt.log_dir + datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.makedirs(log_dir)

appr = approach.App(model, netG, netD, log_dir, opt.outf, niter=opt.niter, batchSize=opt.batchSize,
                    imageSize=opt.imageSize, nz=int(opt.nz), nb_label=nb_label, cuda=torch.cuda.is_available(), beta1=opt.beta1,
                    lr_D=opt.lr_D, lr_G=opt.lr_G, lamb_G=opt.lamb_G,
                    reinit_D=opt.reinit_D, lambda_adv=opt.lambda_adv, lambda_wassersten=opt.lambda_wasserstein, dataset=opt.dataset, store_model = opt.store_models)


for t in range(10):
    test_acc_task, conf_matrixes_task, mask_G = appr.train(data, t, smax_g=opt.smax_g,use_aux_G=opt.aux_G)
