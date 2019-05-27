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
import copy
import numpy as np
import math
from itertools import chain
from utils.utils import accuracy
from utils.utils import AverageMeter
from utils.utils import weights_init
from utils.inception import InceptionV3

from utils.inception_score import inception_score

from utils.fid import get_activations, calculate_frechet_distance


from utils.logger import Logger
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.autograd import Variable
from utils.folder import ImageFolder

class App(object):
    def __init__(self, model, netG, netD, log_dir, outf, out_models, niter=100, batchSize=64, imageSize=64, nz=100, nb_label=10, cuda=True, beta1=0.5,
        lr_D = 0.00002, lr_G=0.0002, lamb_G=1, reinit_D=False, lambd_adv=1, lambda_wassersten=10, dataroot_test=None, dataroot=None,store_model=False, calc_fid_imnet=False, class_idx=[1,15,29,45,59,65,81,89,90,99]):
        self.lambda_wassersten = lambda_wassersten
        self.outf_models = out_models
        self.calc_fid = calc_fid_imnet
        self.idx = class_idx
        self.dataroot_test=dataroot_test
        self.dataroot = dataroot
        self.store_model = store_model
        self.model =model
        self.netG = netG
        self.netD = netD
        self.log_dir = log_dir
        self.writer = Logger(log_dir)
        self.acc_writers = []
        self.reinit_D = reinit_D
        self.lambd_adv = lambd_adv
        self.block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.inc = InceptionV3([self.block_idx])
        self.outf = outf
        self.niter = niter
        self.nb_label = nb_label
        self.nz = nz
        self.cuda = cuda
        self.lr_D = lr_D
        self.lr_G = lr_G
        self.beta1 = beta1
        self.lamb_G = lamb_G
        self.c_criterion = nn.CrossEntropyLoss()
        self.batchSize = batchSize
        self.imageSize = imageSize
        input_ = torch.FloatTensor(batchSize, 3, imageSize, imageSize)
        input_f = torch.FloatTensor(batchSize, 3, imageSize, imageSize)
        noise = torch.FloatTensor(batchSize, nz, 1, 1)
        s_label = torch.FloatTensor(batchSize)
        s_label_fake = torch.FloatTensor(batchSize)
        c_label = torch.LongTensor(batchSize)
        c_label_test = torch.LongTensor(batchSize)

        self.image_distance = torch.nn.modules.loss.L1Loss()
        self.cuda0 = self.netD.device
        self.cuda2 = self.netG.device
        self.netG.device = self.cuda2

        if cuda:
            self.inc.cuda(self.cuda0)
            self.netD.cuda(self.cuda0)
            self.netG.cuda(self.cuda2)
            self.c_criterion.cuda(self.cuda0)
            self.image_distance.cuda(self.cuda0)
            input_, s_label, input_f = input_.cuda(self.cuda0), s_label.cuda(self.cuda0), input_f.cuda(self.cuda0)
            self.s_label_fake = s_label_fake.cuda(self.cuda0)
            c_label = c_label.cuda(self.cuda0)
            self.c_label_test = c_label_test.cuda(self.cuda0)
            noise = noise.cuda(self.cuda2)

        self.input_ = Variable(input_)
        self.input_f = Variable(input_f)
        self.s_label = Variable(s_label)
        self.c_label = Variable(c_label)
        self.noise = Variable(noise)

        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr_D, betas=(self.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr_G, betas=(self.beta1, 0.999))
        self.mask_pre_G = None
        self.mask_back_G = None
        self.n_reserver_prev = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
        self.global_step = 0
        self.unique_classes = []
        self.total_n_params_init = self.netG.total_size_n_params()
        self.writer.scalar_summary('Total Network size', self.netG.total_size(), 0)
        self.writer.scalar_summary('Total Network size (N params)', self.total_n_params_init, 0)

        self.X_act = []
        self.unique_classes = []
        return




    def train(self, dataloader, dataset_train, t, thres_cosh=50, clipgrad=10000, smax_g=1e5, use_aux_G=False):
        self.netD.train()
        self.netG.train()
        n_params = self.netG.total_size_n_params() #self.total_size()
        lamb_G = self.lamb_G * (n_params / self.total_n_params_init) #self.size_init)
        print("lamb_G", lamb_G)
        #init writers
        log_dir_task = self.log_dir+"/Acc. Task "+str(t)
        self.acc_writers.append(Logger(log_dir_task))
        test_acc_task=[]

        self.unique_classes.append(list(range(0+(t*10), 10+(t*10))))
        print('*'*100)
        print("Training on task: ", t)
        print('*'*100)
        print(len(self.unique_classes[t]))
        if t>0:
            old_weights = self.netD.aux_linear.weight.data.clone()
            print(old_weights.shape)
            self.netD.aux_linear = nn.Linear(self.netD.feats, len(self.unique_classes[t])*(t+1)).cuda(self.cuda0)
            self.netD.aux_linear.apply(self.model.weights_init_g)
            self.netD.aux_linear.weight.data[:len(self.unique_classes[t])*t,:].copy_(old_weights)
        for _ in range(len(self.unique_classes[t])):
            self.netG.last.append(self.model.Plastic_Conv2d(self.netG.block3.cap_conv2[t], 3, kernel_size=3, padding=1).cuda(self.cuda2))
        if self.cuda:
           self.netG.cuda(self.cuda2)
           self.netD.cuda(self.cuda0)

        self.netD.disc_linear.reset_parameters()
        if self.reinit_D:
            self.netD.apply(weights_init)

        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr_D, betas=(self.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr_G, betas=(self.beta1, 0.999))

        if t > 0:
            data_past = []
            self.netG.eval()
            with torch.no_grad():
                print("Generating datasets")
                numb_samples = int(len(dataloader.dataset) / 10)
                for t_past in range(t):
                    data_past.append({"train": {"x": None, "y": None}, "test": {"x": None, "y": None}})
                    task_t_past = torch.autograd.Variable(torch.LongTensor([t_past]).cuda(self.cuda2))
                    for counter, c in enumerate(self.unique_classes[t_past]):
                        print("Generating class ",c)
                        self.c_label.data.resize_(numb_samples).fill_(c)
                        _, radom_label = self.generate_noise(numb_samples, self.c_label.data.cpu().numpy())
                        img_gen, _ = self.netG(self.noise, task_t_past, smax_g, past_generation=True, lables = self.c_label)


                        if data_past[t_past]["train"]["x"] is None:
                            data_past[t_past]["train"]["x"] = img_gen.detach().cpu().data.clone()
                        else:
                            data_past[t_past]["train"]["x"] = torch.cat(
                                (data_past[t_past]["train"]["x"], img_gen.detach().cpu().data.clone()))

                        if data_past[t_past]["train"]["y"] is None:
                            data_past[t_past]["train"]["y"] = torch.LongTensor(self.c_label.data.cpu()).cpu().data.clone()
                        else:
                            data_past[t_past]["train"]["y"] = torch.cat(
                                (data_past[t_past]["train"]["y"], torch.LongTensor(self.c_label.data.cpu()).cpu().data.clone()))

            self.netG.train()
            print("*" * 100)
            print("Generating datasets finished")
            print("*" * 100)

        task = torch.autograd.Variable(torch.LongTensor([t]).cuda(self.cuda2))
        _ = self.valid(t, 0, self.netD)


        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=len(dataset_train),
                                                              shuffle=True, num_workers=int(2))
        if self.calc_fid:
            d,l = next(iter(dataloader_train))
            for d, l  in dataloader_train:
                self.input_.data.resize_(d.size()).copy_(d)
                a = torch.nn.functional.interpolate(self.input_, scale_factor=2)
                self.X_act.append(
                   get_activations(a.cpu().data.numpy(), self.inc, batch_size=self.batchSize, dims=2048, cuda=True, device=self.cuda0))
                del (a,d,l)


        try:
            for epoch in range(self.niter):
                #random mix
                for t_past in range(t):      
                    idxx = np.random.permutation(data_past[t_past]["train"]["y"].shape[0])
                    data_past[t_past]["train"]["x"] = data_past[t_past]["train"]["x"][idxx]
                    data_past[t_past]["train"]["y"] = data_past[t_past]["train"]["y"][idxx]
                s_g_max=(smax_g-1/smax_g)*(epoch+1)/self.niter+1/smax_g
                self.write_log_epoch_start(t, epoch, s_g_max)
                for iteratin, data in enumerate(dataloader, 0):


                    ###########################
                    # (1) Update D network
                    ###########################
                    s_g=(s_g_max-1/s_g_max)*iteratin/(int(math.floor(len(dataloader.dataset) / dataloader.batch_size)+1 ))+1/s_g_max
                    # train with real
                    self.netD.zero_grad()
                    img, label = data
                    if img.shape[0]>1:
                        label +=(t*len(self.unique_classes[t]))
                        batch_size = img.size(0)
                        aux_img = img
                        aux_label = label
                        if t>0:
                            for t_past in range(t):
                                aux_img = torch.cat((aux_img, data_past[t_past]['train']['x'][iteratin*self.batchSize:(iteratin*self.batchSize+self.batchSize)]))
                                aux_label = torch.cat((aux_label, data_past[t_past]['train']['y'][iteratin*self.batchSize:(iteratin*self.batchSize+self.batchSize)]))
                        aux_batch_size = aux_img.size(0)

                        idx = np.random.permutation(aux_label.shape[0])
                        aux_img = aux_img[idx]
                        aux_label = aux_label[idx]

                        for bb in range(0, aux_batch_size, batch_size):
                            img_b = aux_img[bb:bb+batch_size]
                            bb_label = aux_label[bb:bb+batch_size]
                            self.input_.data.resize_(img_b.size()).copy_(img_b.detach())
                            self.c_label.data.resize_(img_b.size(0)).copy_(bb_label) # fill with real class labels
                            #print(self.c_label)
                            _, c_output = self.netD(self.input_)
                            c_errD_real  = self.c_criterion(c_output, self.c_label)
                            c_errD_real.backward()
                            self.optimizerD.step()
                            self.netD.zero_grad()

                        self.netD.zero_grad()
                        batch_size = img.size(0)
                        self.input_.resize_(img.size()).copy_(img)
                        self.c_label.data.resize_(batch_size).copy_(label)

                        s_output, _ = self.netD(self.input_)
                        D_x = s_output.mean()
                        s_errD_real = -D_x

                        errD_real = s_errD_real
                        s_errD_real.backward()

                        #### train with fake #####
                        n_fake = batch_size
                        _, radom_label = self.generate_noise(n_fake, self.c_label.data.cpu().numpy())
                        fake, _ = self.netG(self.noise, task, s_g, lables=self.c_label)
                        self.input_f.data.resize_(fake.size()).copy_(fake.detach())
                        s_output, _ = self.netD(self.input_f)
                        s_errD_fake = s_output.mean()
                        errD_fake = s_errD_fake
                        s_errD_fake.backward()
                        gradient_penalty = self.calc_gradient_penalty(self.netD, self.input_, self.input_f, batch_size)
                        gradient_penalty.backward()
                        errD = errD_fake - errD_real + gradient_penalty
                        #torch.nn.utils.clip_grad_norm_(self.netD.parameters(), clipgrad)
                        self.optimizerD.step()

                        ###########################
                        # (2) Update G network
                        ###########################
                        self.netG.zero_grad()
                        fake, masks_G = self.netG(self.noise, task, s_g,lables=self.c_label)
                        s_output, c_output = self.netD(fake.to(device=self.cuda0))
                        source_l, mask_reg_l, _ , reconstruction= self.criterion(s_output, masks_G, lamb_G, real=self.input_, fake=self.input_f)
                        c_errG = self.c_criterion(c_output, self.c_label)

                        #tracking gradients for different losses
                        step = (int(math.floor(iteratin/self.batchSize))) + (int(math.floor(len(dataloader.dataset) / dataloader.batch_size)+1 )* epoch)
                        errG = -(source_l) + mask_reg_l.to(device=self.cuda0) + reconstruction
                        if use_aux_G:
                            errG += c_errG

                        errG.backward()
                        if t>0:
                            for n,p in self.netG.named_parameters():
                                if n in self.mask_back_G:
                                    if p.grad is not None:
                                        p.grad.data*=self.mask_back_G[n]

                        # Compensate embedding gradients
                        for n,p in self.netG.named_parameters():
                            if "ec." in n:#if n.startswith('e'):
                                num=torch.cosh(torch.clamp(s_g*p.data,-thres_cosh,thres_cosh))+1
                                den=torch.cosh(p.data)+1
                                if p.grad is not None:
                                    p.grad.data *= s_g_max/s_g*num/den

                        # Apply step
                        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), clipgrad)
                        self.optimizerG.step()
                        print('|[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)):  %.4f / %.5f'
                              % (epoch, self.niter, iteratin, len(dataloader.dataset)/dataloader.batch_size,
                                 errD.data.item(), errG.data.item(), D_x, s_errD_fake, source_l))



                if (epoch % 2) == 0:
                    ## test D
                    self.netG.eval()
                    with torch.no_grad():
                        test_accs = self.valid(t, epoch, self.netD)
                        print("-" * 100)
                        fid = []
                        inc = []
                        if (epoch % 10) == 0:
                            vutils.save_image(aux_img,
                                              '%s/real_samples_task%d_epoch_%d.png' % (self.outf, t, epoch),normalize=True)
                            for u in range(t):
                                lables_noise = torch.FloatTensor(list(chain(*([x] * 20 for x in range(
                                    min(self.unique_classes[u]),
                                    max(self.unique_classes[u]) + 1)))))  # .cuda()
                                self.c_label.data.resize_(lables_noise.shape[0]).copy_(lables_noise)
                                _, _ = self.generate_noise(lables_noise.shape[0],
                                                               self.c_label.data.cpu().numpy())
                                fake, _ = self.netG(self.noise, u, smax_g, past_generation=True,  lables = self.c_label)#s_g_max)
                                vutils.save_image(fake.data,
                                                  '%s/fake_samples_from_%d_task_%d_epoch_%03d.png' % (
                                                  self.outf, u, t, epoch), normalize=True)

                                # calculate FID score
                                if self.calc_fid:
                                    b = torch.nn.functional.interpolate(fake, scale_factor=2)
                                    X_act = self.X_act[u]
                                    recon_act = get_activations(b.cpu().data.numpy(), self.inc, batch_size=self.batchSize,
                                                                dims=2048, cuda=True, device=self.cuda0)

                                    X_act_mu = np.mean(X_act, axis=0)
                                    recon_act_mu = np.mean(recon_act, axis=0)
                                    X_act_sigma = np.cov(X_act, rowvar=False)
                                    recon_act_sigma = np.cov(recon_act, rowvar=False)
                                    fid.append(
                                        calculate_frechet_distance(recon_act_mu,recon_act_sigma,X_act_mu,X_act_sigma , eps=1e-6))

                            lables_noise = torch.FloatTensor(list(chain(*([x] * 40 for x in
                                                                          range(min(self.unique_classes[t]),
                                                                                max(self.unique_classes[t]) + 1)))))
                            print(lables_noise)


                            self.c_label.data.resize_(lables_noise.shape[0]).copy_(lables_noise)
                            _, radom_label = self.generate_noise(lables_noise.shape[0],
                                                                     self.c_label.data.cpu().numpy())

                            fake, _ = self.netG(self.noise, t, smax_g, lables=self.c_label)  # s_g_max)

                            vutils.save_image(fake.data,
                                              '%s/fake_samples__task_%d_epoch_%03d.png' % (self.outf, t, epoch),
                                              normalize=True)

                            if self.calc_fid:
                                b = torch.nn.functional.interpolate(fake, scale_factor=2)
                                # calculate FID score
                                X_act = self.X_act[t]
                                recon_act = get_activations(b.cpu().data.numpy(), self.inc, batch_size=self.batchSize,
                                                            dims=2048, cuda=True, device=self.cuda0)

                                print("calculating FID")
                                X_act_mu = np.mean(X_act, axis=0)
                                recon_act_mu = np.mean(recon_act, axis=0)
                                X_act_sigma = np.cov(X_act, rowvar=False)
                                recon_act_sigma = np.cov(recon_act, rowvar=False)
                                fid.append(calculate_frechet_distance(recon_act_mu,recon_act_sigma,X_act_mu,X_act_sigma , eps=1e-6))
                                #print("inc", inc)
                                print("fid", fid)
                                del (lables_noise, fake, bb, _)

                    self.netG.train()
                    self.netD.train()
                    self.write_log_epoch_end(t, epoch, s_g, fid, inc) #smax_g, fid, inc)
                if (epoch%100)==0 or epoch==self.niter-1 and self.store_model:
                    # do checkpointing
                    self.store_models(t,epoch)

                self.global_step +=1.
            _ = self.valid(t, epoch, self.netD)
        except KeyboardInterrupt:
            _ = self.valid(t, epoch, self.netD)
            print()

        # Activations mask
        task=torch.autograd.Variable(torch.LongTensor([t]).cuda(self.cuda2))
        masks_G=self.netG.get_total_mask(task,s=smax_g)

        for iii in range(len(masks_G)):
            masks_G[iii][0][masks_G[iii][0] >= 0.5] = 1
            masks_G[iii][0][masks_G[iii][0] < 0.5] = 0
            masks_G[iii][0]=torch.autograd.Variable(masks_G[iii][0].data.clone(),requires_grad=False)
            if masks_G[iii][1] is not None:
                masks_G[iii][1][masks_G[iii][1] >= 0.5] = 1
                masks_G[iii][1][masks_G[iii][1] < 0.5] = 0
                masks_G[iii][1] = torch.autograd.Variable(masks_G[iii][1].data.clone(), requires_grad=False)

        if t==0:
            self.mask_pre_G=masks_G
        else:
            for iiii in range(len(self.mask_pre_G)):
                self.mask_pre_G[iiii][0]=torch.max(self.mask_pre_G[iiii][0],masks_G[iiii][0])
                if masks_G[iiii][1] is not None:
                    self.mask_pre_G[iiii][1] = torch.max(self.mask_pre_G[iiii][1], masks_G[iiii][1])

        current_weight_shapes, newly_used_cap = self.extand_layers(self.mask_pre_G, t, smax_g)
        newly_used_cap = [item for sublist in newly_used_cap for item in sublist]
        _ = self.netG.expand_embeddings(10, t, masks_G)
        masks_G=self.netG.get_total_mask(task.to(self.cuda2),s=smax_g)
        for iii in range(len(masks_G)):
            print("masks_G[iii][0]", self.mask_pre_G[iii][0].shape)
            masks_G[iii][0] = torch.autograd.Variable(masks_G[iii][0].data.clone(), requires_grad=False)
            if masks_G[iii][1] is not None:
                masks_G[iii][1] = torch.autograd.Variable(masks_G[iii][1].data.clone(), requires_grad=False)

            if len(self.mask_pre_G[iii][0].shape) >2:
                self.mask_pre_G[iii][0] = F.pad(self.mask_pre_G[iii][0],
                                             [0, 0, 0, 0,
                                              0, masks_G[iii][0].shape[1] -
                                              self.mask_pre_G[iii][0].shape[1],
                                              0, masks_G[iii][0].shape[0] -
                                              self.mask_pre_G[iii][0].shape[0]],
                                             "constant", 0)

            else:
                self.mask_pre_G[iii][0] = F.pad(self.mask_pre_G[iii][0],
                                                [0, masks_G[iii][0].shape[1] -
                                                 self.mask_pre_G[iii][0].shape[1],
                                                 0, masks_G[iii][0].shape[0] -
                                                 self.mask_pre_G[iii][0].shape[0]],
                                                "constant", 0)

        self.write_log_task_end(t, masks_G, newly_used_cap, smax_g)  # s_g_max) #current_weight_shapes, s_g_max)
        self.mask_back_G={}
        for n,_ in self.netG.named_parameters():
            if n.startswith("block1"):
                vals=self.netG.block1.get_view_for(n,self.mask_pre_G[1:4])
            elif n.startswith("block2"):
                vals=self.netG.block2.get_view_for(n,self.mask_pre_G[4:7])
            elif n.startswith("block3"):
                vals=self.netG.block3.get_view_for(n,self.mask_pre_G[7:10])
            else:
                vals=self.netG.get_view_for(n,self.mask_pre_G[0])
            if vals is not None:
                self.mask_back_G[n]=1-vals

        return test_acc_task, None, masks_G


    def store_models(self, t, epoch):
        bk_1_1 = copy.deepcopy(self.netG.block1.conv1.ec_past)
        bk_1_2 = copy.deepcopy(self.netG.block1.conv2.ec_past)
        bk_1_3 = copy.deepcopy(self.netG.block1.shortcut_conv.ec_past)
        self.netG.block1.conv1.ec_past = self.netG.block1.conv1.ec_past.to_dense()
        self.netG.block1.conv2.ec_past = self.netG.block1.conv2.ec_past.to_dense()
        self.netG.block1.shortcut_conv.ec_past = self.netG.block1.shortcut_conv.ec_past.to_dense()

        bk_2_1 = copy.deepcopy(self.netG.block2.conv1.ec_past)
        bk_2_2 = copy.deepcopy(self.netG.block2.conv2.ec_past)
        bk_2_3 = copy.deepcopy(self.netG.block2.shortcut_conv.ec_past)
        self.netG.block2.conv1.ec_past = self.netG.block2.conv1.ec_past.to_dense()
        self.netG.block2.conv2.ec_past = self.netG.block2.conv2.ec_past.to_dense()
        self.netG.block2.shortcut_conv.ec_past = self.netG.block2.shortcut_conv.ec_past.to_dense()

        bk_3_1 = copy.deepcopy(self.netG.block3.conv1.ec_past)
        bk_3_2 = copy.deepcopy(self.netG.block3.conv2.ec_past)
        bk_3_3 = copy.deepcopy(self.netG.block3.shortcut_conv.ec_past)
        self.netG.block3.conv1.ec_past = self.netG.block3.conv1.ec_past.to_dense()
        self.netG.block3.conv2.ec_past = self.netG.block3.conv2.ec_past.to_dense()
        self.netG.block3.shortcut_conv.ec_past = self.netG.block3.shortcut_conv.ec_past.to_dense()

        fc_ = copy.deepcopy(self.netG.fc1.ec_past)
        self.netG.fc1.ec_past = self.netG.fc1.ec_past.to_dense()

        netD_dev = self.netD.device
        netG_dev = self.netG.device
        self.netD.device = None
        self.netG.device = None
        self.netG.fc1.device = None
        self.netG.block1.device = None
        self.netG.block2.device = None
        self.netG.block3.device = None
        self.netG.block1.conv1.device=None
        self.netG.block1.conv2.device = None
        self.netG.block1.shortcut_conv.device = None
        self.netG.block2.conv1.device = None
        self.netG.block2.conv2.device = None
        self.netG.block2.shortcut_conv.device = None
        self.netG.block3.conv1.device = None
        self.netG.block3.conv2.device = None
        self.netG.block3.shortcut_conv.device = None
        for l in self.netG.last:
            l.ec_past = None
            l.device = None
        torch.save(self.netG, '%s/netG_task_%d_epoch_%d.pth' % (self.outf_models, t, epoch))
        torch.save(self.netD, '%s/netD_task_%d_epoch_%d.pth' % (self.outf_models, t, epoch))

        self.netG.fc1.device = netG_dev
        self.netG.block1.device = netG_dev
        self.netG.block2.device = netG_dev
        self.netG.block3.device = netG_dev
        self.netG.block1.conv1.device = netG_dev
        self.netG.block1.conv2.device = netG_dev
        self.netG.block1.shortcut_conv.device = netG_dev
        self.netG.block2.conv1.device = netG_dev
        self.netG.block2.conv2.device = netG_dev
        self.netG.block2.shortcut_conv.device = netG_dev
        self.netG.block3.conv1.device = netG_dev
        self.netG.block3.conv2.device = netG_dev
        self.netG.block3.shortcut_conv.device = netG_dev
        self.netD.device=netD_dev
        self.netG.device = netG_dev
        for l in self.netG.last:
            l.device = netG_dev

        self.netG.block1.conv1.ec_past = bk_1_1.to(self.netG.device)
        self.netG.block1.conv2.ec_past = bk_1_2.to(self.netG.device)
        self.netG.block1.shortcut_conv.ec_past = bk_1_3.to(self.netG.device)
        self.netG.block2.conv1.ec_past = bk_2_1.to(self.netG.device)
        self.netG.block2.conv2.ec_past = bk_2_2.to(self.netG.device)
        self.netG.block2.shortcut_conv.ec_past = bk_2_3.to(self.netG.device)
        self.netG.block3.conv1.ec_past = bk_3_1.to(self.netG.device)
        self.netG.block3.conv2.ec_past = bk_3_2.to(self.netG.device)
        self.netG.block3.shortcut_conv.ec_past = bk_3_3.to(self.netG.device)
        self.netG.fc1.ec_past = fc_.to(self.netG.device)

        del (bk_1_1, bk_1_2, bk_1_3, bk_2_1, bk_2_2, bk_2_3, bk_3_1, bk_3_2, bk_3_3, fc_)
        del (netD_dev, netG_dev)




    def calc_gradient_penalty(self, netD, real_data, fake_data, BATCH_SIZE):
        LAMBDA = self.lambda_wassersten
        DIM = 32
        alpha = torch.rand(BATCH_SIZE, 1)
        alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement()/BATCH_SIZE)).contiguous()
        alpha = alpha.view(BATCH_SIZE, 3, DIM, DIM).cuda(self.cuda0)

        fake_data = fake_data.view(BATCH_SIZE, 3, DIM, DIM)
        interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

        interpolates = interpolates.cuda(self.cuda0)#.to(device)
        interpolates.requires_grad_(True)

        disc_interpolates, _ = netD(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(self.cuda0),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty



    def generate_noise(self, batch_size, label):
        self.noise.data.resize_(batch_size, self.nz, 1, 1)
        self.noise.data.normal_(0, 1)
        noise_ = np.random.normal(0, 1, (batch_size, self.nz))
        label_onehot = np.zeros((batch_size, self.nb_label))
        label_onehot[np.arange(batch_size), label.astype(int)] = 1.
        noise_[np.arange(batch_size), :self.nb_label] = label_onehot[np.arange(batch_size)]

        noise_ = (torch.from_numpy(noise_))
        noise_ = noise_.resize_(batch_size, self.nz, 1, 1)
        self.noise.data.copy_(noise_)
        return self.noise, label

    def extand_layers(self, mask_pre_G, t, smax_g):
        #addig neurons to keep free capacity constant
        extantion = []
        for layer_n in range(len(mask_pre_G)):
            n_weight = int(torch.sum(mask_pre_G[layer_n][0]==1).data.cpu().numpy()) - self.n_reserver_prev[layer_n][0]
            self.n_reserver_prev[layer_n][0] += n_weight
            n_bias = 0
            if mask_pre_G[layer_n][1] is not None:
                n_bias = int(torch.sum(mask_pre_G[layer_n][1]==1).data.cpu().numpy()) - self.n_reserver_prev[layer_n][1]
            self.n_reserver_prev[layer_n][1] += n_bias
            extantion.append([n_weight, n_bias])
        current_weight_shapes = self.netG.extand(t,extantion, smax_g)

        return current_weight_shapes, extantion

    def write_log_epoch_start(self, t, epoch, smax_g):
        task=torch.autograd.Variable(torch.LongTensor([0]).cuda(self.cuda2))#,volatile=False)
        masks_G = self.netG.get_total_mask(task,s=smax_g)
        cap_string = "Mask capacity G: "
        for layer_n in range(len(masks_G)):
            cap = torch.sum(masks_G[layer_n][0]).cpu().data.numpy() / np.prod(masks_G[layer_n][0].size()).item()
            cap_string += " "+str(cap)
            self.writer.scalar_summary('task_%s/L_%s_mask_capacities'%(t, layer_n), cap, epoch)
            del(cap)
        print(cap_string)

    def write_log_epoch_end(self, t, epoch, smax_g, fid, inc):
        task = torch.autograd.Variable(torch.LongTensor([0]).cuda(self.cuda2))  # ,volatile=False)
        masks_G = self.netG.get_total_mask(task, s=smax_g)
        #cap_string = "Mask capacity G: "
        for layer_n in range(len(masks_G)):
            cap = torch.sum(masks_G[layer_n][0]).cpu().data.numpy() / np.prod(masks_G[layer_n][0].size()).item()
            #cap_string += " " + str(cap)
            self.writer.histo_summary("task_%s/L_%s_mask_distribution" % (t, layer_n),
                                      masks_G[layer_n][0].squeeze(0).cpu().data.numpy(), epoch)
        for ttt in range(len(fid)):
            self.acc_writers[ttt].scalar_summary("FID", 100. * fid[ttt], self.global_step)
            #self.acc_writers[ttt].scalar_summary("INC", inc[ttt], self.global_step)
        del(masks_G)
        #print(cap_string)

    def write_log_task_end(self, t, masks_G, newly_used_cap, smax_g):
        n_free = 0
        reused = 0
        used_ever = 0
        used_last_task = 0
        l_reu_sum = 0
        n_total = 0
        for layer_n in range(len(masks_G)):
            n_free += torch.sum(self.mask_pre_G[layer_n][0]==0)
            n_total += np.prod(self.mask_pre_G[layer_n][0].size())
            layer_mask_acc = masks_G[layer_n][0].data.clone()
            n_total_l = int(masks_G[layer_n][0].shape[1])
            self.writer.scalar_summary('Total capacity L_%s' % (layer_n), n_total_l, t)
            for tt in range(t):
                task_prev=torch.autograd.Variable(torch.LongTensor([tt]).cuda(self.cuda2))
                mask_prev = self.netG.get_total_mask(task_prev,s=smax_g)
                layer_mask_acc[layer_mask_acc>0] += mask_prev[layer_n][0][layer_mask_acc>0]

            l = layer_mask_acc.data.cpu().numpy()
            l_reu = (np.mean(l[l>0] ))
            l_reu_sum+=l_reu
            reused += torch.sum(layer_mask_acc>1)
            used_ever += torch.sum(layer_mask_acc>0)
            used_last_task += torch.sum(masks_G[layer_n][0]>0)
        #log used capacity new
        self.writer.scalar_summary('Newly blocked capacity(% of free)', (sum(newly_used_cap)/n_free.data.cpu().numpy())*100., t)
        self.writer.scalar_summary('Free parameters (N)', n_free, t)
        self.writer.scalar_summary('Newly blocked capacity(absolute)', sum(newly_used_cap), t)
        self.writer.scalar_summary('Parameters used for task (N)', used_last_task.data.cpu().numpy(), t)
        self.writer.scalar_summary('Total Network size', self.netG.total_size(), t+1)
        self.writer.scalar_summary('Total Network size (N params)', self.netG.total_size_n_params(), t+1)


    def valid(self, t_max, epoch, model):
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to evaluate mode
        model.eval()
        #eval on each class seperately
        acc_av = 0
        acc_av5 = 0
        with torch.no_grad():
            for t_past in range(t_max+1):
                idx_ = [i + (t_past * 100) for i in self.idx]
                top1.reset()
                top5.reset()
                print(t_past)
                dataset_test = ImageFolder(
                    root=self.dataroot_test,
                    transform=transforms.Compose([
                        transforms.Resize(self.imageSize),
                        transforms.CenterCrop(self.imageSize),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]),
                    classes_idx=(idx_),
                )
                val_loader_task = torch.utils.data.DataLoader(dataset_test, batch_size=self.batchSize,
                                             shuffle=True, num_workers=int(1))
                for i, (input, target) in enumerate(val_loader_task):
                    #if args.gpu is not None:
                    input = input.cuda(self.cuda0)
                    target = target.cuda(self.cuda0) + (t_past*len(self.unique_classes[t_past]))
                    #print(target)
                    self.c_label.data.resize_(target.shape[0]).copy_(target)

                    # compute output
                    _, output = model(input)
                    output = torch.nn.functional.softmax(output, dim=1)
                    topk = ([1, 5]) #min(t_max+len(self.unique_classes[t_past]),5)])
                    acc1, acc5 = accuracy(output, target, topk=topk)
                    top1.update(acc1) #, input.size(0))
                    top5.update(acc5) #, input.size(0))

                acc_av +=top1.avg
                print('Test: {}, Acc@1 {top1.val[0]:.3f} ({top1.avg[0]:.3f})), Acc@5 {top5.val[0]:.3f} ({top5.avg[0]:.3f}))'.format( #Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                t_past, top1=top1, top5=top5))
                self.acc_writers[t_past].scalar_summary("Accuracy top 1", top1.avg[0], self.global_step)
                self.acc_writers[t_past].scalar_summary("Accuracy top 5", top5.avg[0], self.global_step)

                self.acc_writers[t_past].scalar_summary("Accuracy top 1_val", top1.val[0], self.global_step)
                self.acc_writers[t_past].scalar_summary("Accuracy top 5_val", top5.val[0], self.global_step)
                acc_av5 +=top5.avg

            self.writer.scalar_summary("Average_Acc. top 1", acc_av / (t_max+1), self.global_step)
            self.writer.scalar_summary("Average_Acc. top 5", acc_av5 / (t_max + 1), self.global_step)
        model.train()
        return top1.avg

    def criterion(self, y_hat, masks, lamb_G, real = None, fake = None):
        reg=0
        count=0

        if self.mask_pre_G is not None:
            for m,mp in zip(masks,self.mask_pre_G):
                aux=1-mp[0]
                reg+=(m[0]*aux).sum()
                count+=aux.sum()
        else:
            for m in masks:
                reg+=m[0].sum()
                count+=np.prod(m[0].size()).item()
        reg/=count
        l_reconstr = self.image_distance(fake, real)
        return self.lambd_adv * (y_hat.mean()), lamb_G * reg, reg, l_reconstr



