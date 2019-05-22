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
from utils.logger import Logger
import torch
import pickle
import torch.nn as nn
import torch.nn.parallel
from itertools import chain
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.autograd as autograd
from utils.utils import weights_init


class App(object):
    def __init__(self, model, netG, netD, log_dir, outf, niter=100, batchSize=64, imageSize=64, nz=100, nb_label=10,
                 cuda=True, beta1=0.5, lr_D=0.00002, lr_G=0.0002, lamb_G=1, reinit_D=False, old_samples_G_lambda=1,
                 lambda_adv=1, lambda_wassersten=10, dataset="mnist", device=None, store_model=False):
        self.store_model = store_model
        self.dataset = dataset
        self.lambda_adv = lambda_adv
        self.lambda_wassersten = lambda_wassersten
        self.model = model
        self.netG = netG

        self.netD = netD
        self.log_dir = log_dir
        self.writer = Logger(log_dir)
        self.acc_writers = []
        self.reinit_D = reinit_D
        self.old_samples_G_lambda = old_samples_G_lambda
        self.outf = outf
        self.best_valid_acc = 0 #defaultlist(lambda:0)
        self.best_model_index = None
        self.best_selected_test_acc = 0 #defaultlist(lambda:0)

        self.niter = niter
        self.nb_label = nb_label
        self.nz = nz
        self.device=device

        self.lr_D = lr_D
        self.lr_G = lr_G
        self.beta1 = beta1
        self.n_reserver_prev = [0, 0, 0, 0, 0, 0, 0]

        self.lamb_G = lamb_G
        self.c_criterion = nn.CrossEntropyLoss()
        self.batchSize = batchSize
        self.imageSize = imageSize

        input_ = torch.FloatTensor(batchSize, 3, imageSize, imageSize)

        noise = torch.FloatTensor(batchSize, nz, 1, 1)
        fixed_noise = torch.FloatTensor(batchSize, nz, 1, 1).normal_(0, 1)
        s_label = torch.FloatTensor(batchSize)
        s_label_fake = torch.FloatTensor(batchSize)
        c_label = torch.LongTensor(batchSize)
        c_label_test = torch.LongTensor(batchSize)
        self.image_distance = torch.nn.modules.loss.L1Loss()

        self.real_label = 1
        self.fake_label = 0
        self.one = torch.FloatTensor([1])
        self.mone = self.one * -1

        if cuda:
            self.one = self.one.cuda(self.device)
            self.netD.cuda(self.device)
            self.netG.cuda(self.device)
            self.c_criterion.cuda(self.device)
            input_, s_label = input_.cuda(self.device), s_label.cuda(self.device)
            self.s_label_fake = s_label_fake.cuda(self.device)
            c_label = c_label.cuda(self.device)
            self.c_label_test = c_label_test.cuda(self.device)
            noise, fixed_noise = noise.cuda(self.device), fixed_noise.cuda(self.device)

        self.input_ = Variable(input_)
        self.s_label = Variable(s_label)
        self.c_label = Variable(c_label)
        self.noise = Variable(noise)
        self.fixed_noise = Variable(fixed_noise)
        fixed_noise_ = np.random.normal(0, 1, (batchSize, nz))
        random_label = np.random.randint(0, nb_label, batchSize)
        print('fixed label:{}'.format(random_label))
        random_onehot = np.zeros((batchSize, nb_label))
        random_onehot[np.arange(batchSize), random_label] = 1
        fixed_noise_[np.arange(batchSize), :nb_label] = random_onehot[np.arange(batchSize)]

        fixed_noise_ = (torch.from_numpy(fixed_noise_))
        fixed_noise_ = fixed_noise_.resize_(batchSize, nz, 1, 1)
        self.fixed_noise.data.copy_(fixed_noise_)

        # setup optimizer
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr_D, betas=(self.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr_G, betas=(self.beta1, 0.999))

        self.mask_pre_G = None
        self.mask_back_G = None
        self.mask_histo = []
        self.n_reserver_1_prev = 0
        self.n_reserver_2_prev = 0
        self.n_reserver_3_prev = 0
        self.n_reserver_4_prev = 0
        self.global_step = 0
        self.unique_classes = []
        self.free_size = self.netG.conv1.weight.shape[1] + self.netG.conv2.weight.shape[1] + \
                         self.netG.conv3.weight.shape[1]
        self.writer.scalar_summary('Total capacity Network (size)', self.free_size, 0)
        self.writer.scalar_summary('Total capacity Network (N parametrs)',
                                   np.prod(self.netG.conv1.weight.size()).item() + np.prod(
                                       self.netG.conv2.weight.size()).item() + np.prod(
                                       self.netG.conv3.weight.size()).item(), 0)
        return

    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.lr
        return optim.Adam(self.netD.parameters(), lr=lr, betas=(self.beta1, 0.999)) # torch.optim.SGD(self.model.parameters(), lr=lr)



    def train(self, data, t, thres_cosh=50, thres_emb=6, clipgrad=10000, smax_g=1e5, use_aux_G=False):
        self.mask_histo.append([[], [], []])
        self.best_valid_acc = 0
        self.best_selected_test_acc = 0
        self.netD.train()
        self.netG.train()

        lr_D = self.lr_D
        total_size = self.netG.conv1.weight.shape[1] + self.netG.conv2.weight.shape[1] + self.netG.conv3.weight.shape[1]
        lamb_G = self.lamb_G * (total_size / self.free_size)
        print("lamb_G", lamb_G)

        # init writers for test accuracies
        log_dir_task = self.log_dir + "/Acc. Task " + str(t)
        self.acc_writers.append(Logger(log_dir_task))
        test_acc_task = []
        data_train_x = data[t]['train']['x'].data.clone()
        data_train_y = data[t]['train']['y'].data.clone()

        print('*' * 100)
        print("Training on task: ", t)
        print('*' * 100)
        self.unique_classes.append(torch.unique(data_train_y))

        if t > 0:
            old_weights = self.netD.aux_linear.weight.data.clone()
            print(old_weights.shape)
            self.netD.aux_linear = nn.Linear(self.netD.output_size, old_weights.shape[0] + len(
                self.unique_classes[t])).cuda(self.device)
            self.netD.aux_linear.apply(weights_init)
            self.netD.aux_linear.weight.data[:old_weights.shape[0], :].copy_(copy.copy(old_weights.data.clone()))

        for c in range(len(self.unique_classes[t])):  # if t == len(self.netG.last):
            self.netG.last.append(self.model.Plastic_ConvTranspose2d(self.netG.cap_conv3[t], self.netG.nc, 4, 2, 1,
                                                                     bias=False).cuda(self.device))
        if t>0:
            self.netD.disc_linear.reset_parameters()
            self.netD.aux_linear.reset_parameters()

        if self.reinit_D and t>0:
            self.netD.apply(weights_init)
            self.netD.aux_linear.reset_parameters()
            self.netD.disc_linear.reset_parameters()
        self.optimizerD =  self._get_optimizer(lr_D) #optim.Adam(self.netD.parameters(), lr=lr_D, betas=(self.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr_G, betas=(self.beta1, 0.999))

        if t > 0:
            print("Generating datasets")
            self.netG.eval()
            numb_samples = int(data[t]['train']['y'].shape[0])
            print(numb_samples)
            for t_past in range(t):
                r = np.arange(numb_samples)
                r = torch.LongTensor(r)  # .cuda()
                data[t_past]["train"]["x"] = None
                data[t_past]["train"]["y"] = None
                for c in self.unique_classes[t_past]:
                    print(c)
                    for i in range(0, len(r), self.batchSize):
                        if i + self.batchSize <= len(r):
                            b = r[i:i + self.batchSize]
                        else:
                            b = r[i:]
                        self.c_label.data.resize_(len(b)).fill_(c)
                        noise, radom_label = self.generate_noise(t_past, len(b), self.c_label.data.cpu().numpy())
                        if data[t_past]["train"]["y"] is None:
                            data[t_past]["train"]["y"] = torch.LongTensor(radom_label).cpu().data.clone()
                        else:
                            data[t_past]["train"]["y"] = torch.cat(
                                (data[t_past]["train"]["y"], torch.LongTensor(radom_label).cpu().data.clone()))

                        img_gen, _ = self.netG(noise, t_past, self.c_label, smax_g)
                        if data[t_past]["train"]["x"] is None:
                            data[t_past]["train"]["x"] = img_gen.detach().cpu().data.clone()
                        else:
                            data[t_past]["train"]["x"] = torch.cat(
                                (data[t_past]["train"]["x"], img_gen.detach().cpu().data.clone()))
                idx = np.random.permutation(data[t_past]["train"]["y"].shape[0])
                data[t_past]["train"]["x"] = data[t_past]["train"]["x"][idx]
                data[t_past]["train"]["y"] = data[t_past]["train"]["y"][idx]
            print("*" * 100)
            print("Generating datasets finished")
            self.netG.train()

            print("*" * 100)

        try:
            for epoch in range(self.niter):
                s_g_max = (smax_g - 1 / smax_g) * epoch / self.niter + 1 / smax_g

                print("s_g_max: ", s_g_max)
                self.write_log_epoch_start(t, epoch, smax_g, lamb_G)
                r = np.arange(data_train_x.shape[0])
                r = torch.LongTensor(r)  # .cuda()

                for i in range(0, len(r), self.batchSize):
                    if i + self.batchSize <= len(r):
                        b = r[i:i + self.batchSize]
                    else:
                        b = r[i:]

                    self.netD.zero_grad()
                    self.netD.train()

                    ###########################
                    # (1) Update D network
                    ###########################

                    s_g =  (s_g_max-1/s_g_max)*i/len(r)+1/s_g_max
                    # train with real
                    img, label = data_train_x[b], data_train_y[b]
                    aux_img = img
                    aux_label = label
                    if t > 0:
                        for t_past in range(t):
                            aux_img = torch.cat((aux_img, data[t_past]['train']['x'][b].detach()))
                            aux_label = torch.cat((aux_label, data[t_past]['train']['y'][b]))

                    idx = np.random.permutation(aux_img.shape[0])
                    aux_img = aux_img[idx]
                    aux_label = aux_label[idx]
                    aux_batch_size = aux_img.size(0)
                    loss_G_aux =[]
                    for bb in range(0, aux_batch_size, len(b)):
                        img_b = aux_img[bb:bb + len(b)]
                        bb_label = aux_label[bb:bb + len(b)]
                        self.input_.data.resize_(img_b.size()).copy_(img_b.detach())
                        self.c_label.data.resize_(img_b.size(0)).copy_(bb_label)  # fill with real class labels
                        _, c_output = self.netD(self.input_)

                        c_errD_real = self.c_criterion(c_output, self.c_label)
                        loss_G_aux.append(c_errD_real)
                        c_errD_real.backward()
                        self.optimizerD.step()
                        self.netD.zero_grad()

                    self.netD.zero_grad()
                    batch_size = img.size(0)
                    self.input_.resize_(img.size()).copy_(img)

                    s_output, _ = self.netD(self.input_)
                    D_x = s_output.mean()
                    s_errD_real = -D_x
                    s_errD_real.backward()

                    n_fake = batch_size
                    self.c_label.data.resize_(batch_size).copy_(label)
                    noise, radom_label = self.generate_noise(t, n_fake, self.c_label.data.cpu().numpy())
                    fake, masks_G = self.netG(noise, t, self.c_label, s_g)
                    s_output_fake, _ = self.netD(fake.detach())
                    D_x_fake = s_output_fake.mean()
                    errD_fake = D_x_fake #s_errD_fake  # + c_errD_fake
                    errD_fake.backward()
                    gradient_penalty = self.calc_gradient_penalty(self.netD, self.input_, fake, batch_size)
                    gradient_penalty.backward()
                    errD = errD_fake - s_errD_real + gradient_penalty
                    #torch.nn.utils.clip_grad_norm_(self.netD.parameters(), clipgrad)
                    self.optimizerD.step()

                    ###########################
                    # (2) Update G network
                    ###########################
                    self.netG.zero_grad()
                    noise, radom_label = self.generate_noise(t, n_fake, self.c_label.data.cpu().numpy())
                    fake, masks_G = self.netG(noise, t, self.c_label, s_g)
                    s_output, c_output = self.netD(fake)
                    source_l, mask_reg_l, _ = self.criterion(s_output, masks_G, lamb_G)
                    c_errG = self.c_criterion(c_output, self.c_label)

                    c = (int(math.floor(i / self.batchSize))) + (
                                int(math.floor(data_train_x.shape[0] / self.batchSize) + 1) * epoch)

                    errG = -(source_l)+ mask_reg_l
                    if use_aux_G:
                        errG +=c_errG
                    errG.backward()
                    if t > 0:
                        for n, p in self.netG.named_parameters():
                            if n in self.mask_back_G and p.grad is not None:
                                p.grad.data *= self.mask_back_G[n]

                    # Compensate embedding gradients like in HAT
                    for n, p in self.netG.named_parameters():
                        if "ec." in n:  # .startswith('e'):
                            num = torch.cosh(torch.clamp(s_g * p.data, -thres_cosh, thres_cosh)) + 1
                            den = torch.cosh(p.data) + 1
                            if p.grad is not None:
                                p.grad.data *= s_g_max / s_g * num / den

                    # Apply step
                    torch.nn.utils.clip_grad_norm_(self.netG.parameters(), clipgrad)
                    self.optimizerG.step()
                    self.netG.zero_grad()
                    print('|[%d/%d][%d/%d] Loss_D: %.2f Loss_G: %.2f D(x): %.2f D(G(z)):  %.2f / %.2f'
                              % (epoch, self.niter, i / self.batchSize, data_train_x.shape[0] / self.batchSize,
                                 errD.data.item(), errG.data.item(), s_errD_real, errD_fake, source_l))

                self.write_log_epoch_end(t, epoch, s_g)  # s_g_max)
                if epoch % 2 == 0:
                    self.netG.eval()
                    loss_valid, valid_acc, _ = self.valid(data, t, epoch, self.netD, "valid")
                    loss, test_accs, _ = self.valid(data, t, epoch, self.netD, "test")
                    test_acc_task.append(test_accs)
                    print("-" * 100)
                    if epoch%10 == 0:
                        norm = False
                        if self.dataset == "svhn":
                            norm = True

                        vutils.save_image(aux_img,
                                          '%s/real_samples_task%d_epoch_%d.png' % (self.outf, t, epoch),normalize=norm)
                        lables_noise = torch.FloatTensor(list(chain(*([x] * 40 for x in
                                                                      range(torch.min(self.unique_classes[t]),
                                                                            torch.max(self.unique_classes[t]) + 1)))))

                        self.c_label.data.resize_(lables_noise.shape[0]).copy_(lables_noise)
                        noise, radom_label = self.generate_noise(t, lables_noise.shape[0], self.c_label.data.cpu().numpy())
                        fake, _ = self.netG(noise, t, self.c_label, s_g) #s_g_max)
                        vutils.save_image(fake.data,
                                          '%s/fake_samples__task_%d_epoch_%03d.png' % (self.outf, t, epoch),normalize=norm)

                        if t > 0:
                            for u in range(t + 1):
                                lables_noise = torch.FloatTensor(list(chain(*([x] * 20 for x in
                                                                              range(torch.min(self.unique_classes[u]),
                                                                                    torch.max(self.unique_classes[u]) + 1)))))
                                self.c_label.data.resize_(lables_noise.shape[0]).copy_(lables_noise)
                                noise, _ = self.generate_noise(u, lables_noise.shape[0], self.c_label.data.cpu().numpy())
                                fake, _ = self.netG(noise, u, self.c_label, smax_g)
                                vutils.save_image(fake.data,
                                                  '%s/fake_samples_from_%d_task_%d_epoch_%03d.png' % (
                                                  self.outf, u, t, epoch),normalize=norm)
                    self.netG.train()
                self.global_step += 1
            loss_valid, valid_acc, _ = self.valid(data, t, epoch, self.netD, "valid")
            loss, test_accs, conf_matrixes_task = self.valid(data, t, epoch, self.netD, "test")

            test_acc_task.append(test_accs)

        except KeyboardInterrupt:
            oss_valid, valid_acc, _ = self.valid(data, t, epoch, self.netD, "valid")
            loss, test_accs, conf_matrixes_task = self.valid(data, t, epoch, self.netD, "test")

            test_acc_task.append(test_accs)
            print()

        # Activations mask
        task = torch.autograd.Variable(torch.LongTensor([t]).cuda(self.device))  # ,volatile=False)
        masks_G = self.netG.mask(task, None, s=smax_g)

        for i_ in range(len(masks_G)):
            masks_G[i_][masks_G[i_] >= 0.5] = 1
            masks_G[i_][masks_G[i_] < 0.5] = 0
            masks_G[i_] = torch.autograd.Variable(masks_G[i_].data.clone(), requires_grad=False)
        if t == 0:
            self.mask_pre_G = masks_G
        else:
            for i_ in range(len(self.mask_pre_G)):
                self.mask_pre_G[i_] = torch.max(self.mask_pre_G[i_], masks_G[i_])

        current_weight_shapes, newly_used_cap = self.extand_layers(self.mask_pre_G, t)
        prev_weight_shapes = self.netG.expand_embeddings(1)
        masks_G = self.netG.mask(task, None, s=smax_g)
        for i_ in range(len(masks_G)):
            masks_G[i_] = torch.autograd.Variable(masks_G[i_].data.clone(), requires_grad=False)
            self.mask_pre_G[i_] = F.pad(self.mask_pre_G[i_].view(list(prev_weight_shapes[i_])),
                                        [0, 0, 0, 0,
                                          0, masks_G[i_].view(list(current_weight_shapes[i_])).shape[1] -
                                          self.mask_pre_G[i_].view(list(prev_weight_shapes[i_])).shape[1],
                                          0, masks_G[i_].view(list(current_weight_shapes[i_])).shape[0] -
                                          self.mask_pre_G[i_].view(list(prev_weight_shapes[i_])).shape[0]],
                                         "constant", 0)
            self.mask_pre_G[i_] = self.mask_pre_G[i_].view([1, np.prod(self.mask_pre_G[i_].size()).item()])

        self.write_log_task_end(t, masks_G, newly_used_cap, current_weight_shapes, smax_g)

        self.mask_back_G = {}
        for n, _ in self.netG.named_parameters():
            vals = self.netG.get_view_for(n, self.mask_pre_G)
            if vals is not None:
                self.mask_back_G[n] = 1 - vals
                self.mask_back_G[n][self.mask_back_G[n] < 0.5] = 0
                self.mask_back_G[n][self.mask_back_G[n] >= 0.5] = 1

        #backup model and mask_histo
        if self.store_model:
            torch.save(self.netG, '%s/netG_task_%d_epoch_%d.pth' % (self.outf + "/models", t, epoch))
            torch.save(self.netD, '%s/netD_task_%d_epoch_%d.pth' % (self.outf + "/models", t, epoch))
            with open(self.outf + '/mask_histo/' + str(t) + '.pickle', 'wb') as handle:
               pickle.dump(self.mask_histo, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return self.best_selected_test_acc, conf_matrixes_task, masks_G

    def generate_noise(self, c, batch_size, label):
        self.noise.data.resize_(batch_size, self.nz, 1, 1)
        noise_ = np.random.normal(0, 1, (batch_size, self.nz))
        label_onehot = np.zeros((batch_size, self.nb_label))
        label_onehot[np.arange(batch_size), label.astype(int)] = 1.
        noise_[np.arange(batch_size), :self.nb_label] = label_onehot[np.arange(batch_size)]

        noise_ = (torch.from_numpy(noise_))
        noise_ = noise_.resize_(batch_size, self.nz, 1, 1)
        self.noise.data.copy_(noise_)

        return copy.copy(self.noise), label

    def write_log_epoch_start(self, t, epoch, smax_g, lamb_G):
        # pass
        task = torch.autograd.Variable(torch.LongTensor([t]).cuda(self.device))  # ,volatile=False)
        current_classes = self.unique_classes[t]
        self.c_label.data.resize_(current_classes.shape[0]).copy_(current_classes)

        masks_G = self.netG.mask(task, self.c_label, s=smax_g)
        total_cap = 0
        total_used = 0
        cap_string = "Mask capacity G: "
        print(len(masks_G))
        print(masks_G[0].shape)
        for layer_n in range(len(masks_G)):
            cap = torch.sum(masks_G[layer_n]).cpu().data.numpy() / np.prod(masks_G[layer_n].size()).item()
            n_total_l = int(masks_G[layer_n].shape[1])

            cap_string += " " + str(cap)
            total_cap += n_total_l
            total_used += torch.sum(masks_G[layer_n]).cpu().data.numpy()
            self.writer.histo_summary("task_%s/L_%s_mask_distribution" % (t, layer_n),
                                      masks_G[layer_n].squeeze(0).cpu().data.numpy(), epoch)
            self.mask_histo[t][layer_n].append(masks_G[layer_n].squeeze(0).cpu().data.numpy())

        self.writer.scalar_summary('lamb_G', lamb_G, t)
        print(cap_string)

    def calc_gradient_penalty(self, netD, real_data, fake_data, BATCH_SIZE):
        LAMBDA = self.lambda_wassersten
        DIM = 32
        alpha = torch.rand(BATCH_SIZE, 1)
        alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement() / BATCH_SIZE)).contiguous()
        alpha = alpha.view(BATCH_SIZE, self.netD.nc, DIM, DIM).cuda(self.device)
        interpolates = alpha * real_data.cuda(self.device) + ((1 - alpha) * fake_data.cuda(self.device))

        # if use_cuda:
        interpolates = interpolates.cuda(self.device)
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates, _ = netD(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(self.device), create_graph=True,
                                  retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty

    def write_log_epoch_end(self, t, epoch, smax_g):
        task = torch.autograd.Variable(torch.LongTensor([t]).cuda(self.device))
        current_classes = self.unique_classes[t]
        self.c_label.data.resize_(current_classes.shape[0]).copy_(current_classes)
        masks_G = self.netG.mask(task, self.c_label, s=smax_g)

        total_cap = 0
        total_used = 0
        cap_string = "Mask capacity G: "
        for layer_n in range(len(masks_G)):
            cap = torch.sum(masks_G[layer_n]).cpu().data.numpy() / np.prod(masks_G[layer_n].size()).item()
            cap_string += " " + str(cap)
            n_total_l = int(masks_G[layer_n].shape[1])
            total_cap += n_total_l
            total_used += torch.sum(masks_G[layer_n]).cpu().data.numpy()
            self.writer.histo_summary("task_%s/L_%s_mask_distribution" % (t, layer_n),
                                      masks_G[layer_n].squeeze(0).cpu().data.numpy(), epoch)

        self.writer.scalar_summary('task_%s/Total_used_capacity' % (t), total_used / total_cap, epoch)
        print(cap_string)

    def write_log_task_end(self, t, masks_G, newly_used_cap, current_weight_shapes, smax_g):
        n_free = 0
        reused = 0
        total_cap = 0
        total_used = 0
        used_ever = 0
        used_last_task = 0
        l_reu_sum = 0
        for layer_n in range(len(masks_G)):
            cap = torch.sum(masks_G[layer_n]).cpu().data.numpy() / np.prod(masks_G[layer_n].size()).item()
            n_free += torch.sum(self.mask_pre_G[layer_n] == 0)
            layer_mask_acc = masks_G[layer_n].data.clone()

            n_total_l = int(masks_G[layer_n].shape[1] / (
                        current_weight_shapes[layer_n][0] * current_weight_shapes[layer_n][2] *
                        current_weight_shapes[layer_n][3]))
            total_cap += n_total_l
            total_used += torch.sum(masks_G[layer_n]).cpu().data.numpy()
            self.writer.scalar_summary('task_%s/L_%s_mask_capacities' % (t, layer_n), cap, t)
            self.writer.scalar_summary('Total capacity L_%s' % (layer_n), n_total_l, t)

            for tt in range(t):
                current_classes = self.unique_classes[tt]
                self.c_label.data.resize_(current_classes.shape[0]).copy_(current_classes)
                task = torch.autograd.Variable(torch.LongTensor([tt]).cuda(self.device))  # ,volatile=False)
                mask_prev = self.netG.mask(task, self.c_label, s=smax_g)
                # mask_prev = self.netG.mask(task_prev, self.unique_classes[tt-1],s=smax_g)
                layer_mask_acc[layer_mask_acc > 0] += mask_prev[layer_n][layer_mask_acc > 0]

            l = layer_mask_acc.data.cpu().numpy()
            l_reu = (np.mean(l[l > 0]))
            l_reu_sum += l_reu
            reused += torch.sum(layer_mask_acc > 1)
            used_ever += torch.sum(layer_mask_acc > 0)
            used_last_task += torch.sum(masks_G[layer_n] > 0)
        # self.writer.scalar_summary('Average reusability (N tasks)', l_reu / len(masks_G), t)
        self.writer.scalar_summary('Total capacity Network (size)',
                                   self.netG.conv1.weight.shape[1] + self.netG.conv2.weight.shape[1] +
                                   self.netG.conv3.weight.shape[1], t + 1)
        self.writer.scalar_summary('Total capacity Network (N parametrs)',
                                   np.prod(self.netG.conv1.weight.size()).item() + np.prod(
                                       self.netG.conv2.weight.size()).item() + np.prod(
                                       self.netG.conv3.weight.size()).item(), t + 1)
        self.writer.scalar_summary('Free parameters (N)', n_free, t + 1)

    def extand_layers(self, masks_G, t):
        # Dynamic network expansion - keep free capacity constant for every task
        extantion = []
        for layer_n in range(len(masks_G)):
            n_reserver = int(torch.sum(masks_G[layer_n] == 1).data.cpu().numpy()) - self.n_reserver_prev[layer_n]
            self.n_reserver_prev[layer_n] += n_reserver
            extantion.append(n_reserver)
        current_weight_shapes = self.netG.extand(extantion)

        return current_weight_shapes, extantion

    def accuracy(self, output, target):
        val, max_ = output.max(1)
        hits = (max_ == target).float()
        acc = torch.sum(hits).data.cpu().numpy() / target.shape[0]
        return acc, max_.data.cpu().numpy()

    def valid(self, data, t_max, epoch, net, split="valid"):
        self.netD.eval()
        test_accs = []
        confusion = None  #
        np.zeros((t_max + 1, t_max + 1))
        acc_av = 0
        loss = 0
        correct_labels = []
        predict_labels = []
        with torch.no_grad():
            for tt in range(t_max + 1):
                total_acc = 0
                total_num = 0
                r_valid = np.arange(data[tt][split]['x'].shape[0])
                r_valid = torch.LongTensor(r_valid).cuda(self.device)
                print("-" * 100)
                # true x pred
                for ii in range(0, len(r_valid), self.batchSize):
                    if ii + self.batchSize <= len(r_valid):
                        b_val = r_valid[ii:ii + self.batchSize]
                    else:
                        b_val = r_valid[ii:]
                    img_valid, label_valid = data[tt][split]['x'][b_val], data[tt][split]['y'][b_val]
                    self.input_.data.resize_(img_valid.size()).copy_(img_valid)
                    correct_labels += list(label_valid)
                    _, c_output_valid = net(self.input_)
                    loss += self.c_criterion(c_output_valid, label_valid.cuda(self.device))
                    c_output_valid = torch.nn.functional.log_softmax(c_output_valid, dim=1)
                    acc_, pred = self.accuracy(c_output_valid, label_valid.cuda(self.device))
                    total_acc += acc_
                    total_num += 1
                    predict_labels += list(pred)
                acc = total_acc / total_num
                acc_av += acc
                test_accs.append(acc)
                print('| '+split+' on task:{:d} : acc={:.1f}% |'.format(tt, 100. * acc), end='\n')
                self.acc_writers[tt].scalar_summary("Accuracy_"+split, 100. * acc, self.global_step)
        acc_av = (100. * acc_av) / (t_max + 1)
        self.writer.scalar_summary("Average_Acc. "+split, acc_av, self.global_step)

        if split == "valid":
            if self.best_valid_acc < acc_av:
                self.best_valid_acc = acc_av
                self.best_model_index = self.global_step
                self.writer.scalar_summary("Best acc " + split, acc_av, t_max)
        elif split == "test":
            if self.best_model_index is not None and self.best_model_index == self.global_step:
                self.writer.scalar_summary("Best acc " + split, acc_av, t_max)
                self.best_selected_test_acc = acc_av
            print("*"*100)
            print('| Best selected average acc. after task {:d} sofar: acc={:.1f}% |'.format(t_max, self.best_selected_test_acc), end='\n')
        self.netD.train()
        return loss, acc_av, confusion

    def criterion(self, y_hat, masks, lamb_G):
        reg = 0
        count = 0
        if self.mask_pre_G is not None:
            for m, mp in zip(masks, self.mask_pre_G):
                aux = 1 - mp
                reg += (m * aux).sum()
                count += aux.sum()
        else:
            for m in masks:
                reg += m.sum()
                count += np.prod(m.size()).item()
        reg /= count
        return self.lambda_adv * (y_hat.mean()), lamb_G * reg, reg

