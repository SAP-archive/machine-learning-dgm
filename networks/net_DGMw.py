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
import torch.nn as nn
import torch
import numpy as np
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F
from utils.utils import weights_init, weights_init_g


class Plastic_ConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 groups=1, bias=True,
                 num_tasks=1, out_size=None, batch_size=64, smax=800):
        super(Plastic_ConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                                      output_padding, groups, bias, dilation)
        if bias:
            self.ec = torch.nn.Embedding(num_tasks,
                                         out_channels * ((in_channels * kernel_size * kernel_size) + 1)).cuda()
        else:
            self.ec = torch.nn.Embedding(num_tasks, out_channels * ((in_channels * kernel_size * kernel_size))).cuda()

        self.smax = smax
        self.ec.weight.data.fill_(0)
        self.prev_weight_shape = self.weight.shape

    def forward(self, inputx, mask, d_in, d_out, output_size=None):
        self.prev_weight_shape = self.weight.shape
        output_padding = self._output_padding(inputx, output_size, self.stride, self.padding, self.kernel_size)
        bias = None
        if mask is not None:
            if not self.bias is None:
                bias = self.bias[:d_out] * mask[:, 0].contiguous().view(-1)
                mask_ = mask[:, 1:]
            else:
                mask_ = mask[:, :]
            out = F.conv_transpose2d(inputx, self.weight[:d_in, :d_out, :, :] * mask_.contiguous().view(
                self.weight.data.shape)[:d_in, :d_out, :, :],
                                     bias, self.stride, self.padding, output_padding, groups=self.groups,
                                     dilation=self.dilation)
        else:
            if not self.bias is None:
                bias = self.bias[:d_out]
            out = F.conv_transpose2d(inputx, self.weight[:d_in, :d_out, :, :], bias, self.stride, self.padding,
                                     output_padding, groups=self.groups, dilation=self.dilation)
        return out

    def expand(self, input_channels, out_channels):
        w_old = self.weight.data.clone()
        if self.bias is not None:
            b_old = self.bias.data.clone()
        self.out_channels += out_channels
        self.in_channels += input_channels
        self.weight = Parameter(
            torch.Tensor(self.in_channels, self.out_channels // self.groups, *self.kernel_size).cuda())
        if self.bias is not None:
            self.bias = Parameter(torch.Tensor(self.out_channels))
        self.weight.data.fill_(0)
        self.apply(weights_init)
        self.weight.data[:w_old.shape[0]:, :w_old.shape[1], :, :].copy_(w_old)
        if self.bias is not None:
            self.bias.data[:b_old.shape[0]].copy_(b_old)
        return self.weight.shape

    def expand_embeddings(self, n_new_classes, mask_pre=None):
        ec = self.ec.weight.view([-1] + list(self.prev_weight_shape)).data.clone()
        new_dim = self.out_channels * ((self.in_channels * self.kernel_size[0] * self.kernel_size[1]))
        if self.bias is not None:
            new_dim = self.out_channels * ((self.in_channels * self.kernel_size[0] * self.kernel_size[1]) + 1)
        self.ec = torch.nn.Embedding(ec.shape[0] + n_new_classes, new_dim).cuda()
        self.ec.weight.data.fill_(0)

        if ec.shape[0] > 0:
            self.ec.weight.data[:ec.shape[0], :].fill_(-90)  # for generating old samples do not use newly added parameters
            self.ec.weight.view([-1] + list(self.weight.shape)).data[:ec.shape[0], :ec.shape[1], :ec.shape[2], :,
            :].copy_(ec[:, :, :, :, :])  # but only the reserved once
        return self.prev_weight_shape


class BatchNorm2d(torch.nn.BatchNorm2d):
    def forward(self, input):
        self._check_input_dim(input)
        if self.training:
            momentum = self.momentum
        else:
            momentum = 0.
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, momentum, self.eps)


class netG(nn.Module):
    def __init__(self, nz, ngf, nc, smax, n_classes=1):
        super(netG, self).__init__()

        self.gate = torch.nn.Sigmoid()

        self.nc = nc
        self.nz = nz
        self.ngf = ngf
        self.scalor = 1
        self.smax = smax
        self.ReLU = nn.ReLU(True)
        self.Tanh = nn.Tanh()

        self.conv1 = Plastic_ConvTranspose2d(nz, ngf * 4 * self.scalor, 4, 1, 0, bias=False, num_tasks=n_classes,
                                             smax=smax)
        # self.conv1 = Plastic_ConvTranspose2d(nz, ngf * 8*self.scalor, 3, 1, 0, bias=False, num_tasks=n_classes, smax=smax)

        self.cap_conv1 = [ngf * 4 * self.scalor]
        self.BatchNorms1 = torch.nn.ModuleList()
        self.BatchNorms1.append(torch.nn.BatchNorm2d(ngf * 4 * self.scalor).apply(weights_init))

        self.conv2 = Plastic_ConvTranspose2d(ngf * 4 * self.scalor, ngf * 2 * self.scalor, 4, 2, 1, bias=False,
                                             num_tasks=n_classes, smax=smax)
        self.cap_conv2 = [ngf * 2 * self.scalor]
        self.BatchNorms2 = torch.nn.ModuleList()
        self.BatchNorms2.append(torch.nn.BatchNorm2d(ngf * 2 * self.scalor).apply(weights_init))

        self.conv3 = Plastic_ConvTranspose2d(ngf * 2 * self.scalor, ngf * 1 * self.scalor, 4, 2, 1, bias=False,
                                             num_tasks=n_classes, smax=smax)
        self.cap_conv3 = [ngf * 1 * self.scalor]
        self.BatchNorms3 = torch.nn.ModuleList()
        self.BatchNorms3.append(torch.nn.BatchNorm2d(ngf * 1 * self.scalor).apply(weights_init))

        self.apply(weights_init_g)
        self.last = self.last = torch.nn.ModuleList()

    def extand(self, extention):
        ws_0 = self.conv1.expand(0, math.ceil(
            extention[0] / (self.nz * self.conv1.kernel_size[0] * self.conv1.kernel_size[1])))

        n_in_conv2 = self.conv1.weight.shape[1]
        n_out_conv2 = self.conv2.weight.shape[1]
        n_params_conv2 = n_in_conv2 * n_out_conv2 * self.conv2.kernel_size[0] * self.conv2.kernel_size[1]
        n_add_out_conv2 = max(math.ceil((extention[1] - (n_params_conv2 - np.prod(self.conv2.weight.size()).item())) / (
                    self.conv1.weight.data.shape[1] * self.conv2.kernel_size[0] * self.conv2.kernel_size[1])), 0)

        ws_1 = self.conv2.expand(
            math.ceil(extention[0] / (self.nz * self.conv1.kernel_size[0] * self.conv1.kernel_size[1])),
            n_add_out_conv2)
        n_in_conv3 = self.conv2.weight.shape[1]
        n_out_conv3 = self.conv3.weight.shape[1]
        n_params_conv3 = n_in_conv3 * n_out_conv3 * self.conv3.kernel_size[0] * self.conv3.kernel_size[1]
        n_add_out_conv3 = max(math.ceil((extention[2] - (n_params_conv3 - np.prod(self.conv3.weight.size()).item())) / (
                self.conv2.weight.data.shape[1] * self.conv3.kernel_size[0] * self.conv3.kernel_size[1])), 0)

        ws_2 = self.conv3.expand(n_add_out_conv2, n_add_out_conv3)
        self.cap_conv1.append(self.conv1.weight.shape[1])
        self.cap_conv2.append(self.conv2.weight.shape[1])
        self.cap_conv3.append(self.conv3.weight.shape[1])
        self.BatchNorms1.append(torch.nn.BatchNorm2d(self.conv1.weight.shape[1]).cuda())
        self.BatchNorms2.append(torch.nn.BatchNorm2d(self.conv2.weight.shape[1]).cuda())
        self.BatchNorms3.append(torch.nn.BatchNorm2d(self.conv3.weight.shape[1]).cuda())

        return [ws_0, ws_1, ws_2]

    def expand_embeddings(self, n_new_classes):
        prev_weight_shape_0 = self.conv1.expand_embeddings(n_new_classes)
        prev_weight_shape_1 = self.conv2.expand_embeddings(n_new_classes)
        prev_weight_shape_2 = self.conv3.expand_embeddings(n_new_classes)
        return [prev_weight_shape_0, prev_weight_shape_1, prev_weight_shape_2]

    def forward(self, input, t, lables=None, s=1, t_mix=None):
        task = torch.autograd.Variable(torch.LongTensor([t]).cuda())
        masks = self.mask(task, None, s=s)

        gc1, gc2, gc3 = masks
        # print(input.shape)
        x = self.conv1(input, gc1, self.nz, self.cap_conv1[t])
        x = self.BatchNorms1[t](x)
        x = self.ReLU(x)

        x = self.conv2(x, gc2, self.cap_conv1[t], self.cap_conv2[t])
        x = self.BatchNorms2[t](x)
        x = self.ReLU(x)

        x = self.conv3(x, gc3, self.cap_conv2[t], self.cap_conv3[t])
        x = self.BatchNorms3[t](x)
        x = self.ReLU(x)
        output = self.Tanh(self.last[t](x, None, self.cap_conv3[t], self.nc))
        return output, masks

    def mask(self, t, labels, s=1, test=False):
        gc1 = self.gate(s * self.conv1.ec(t))  # .view(self.conv1.out_channels, (
        # self.conv1.in_channels * self.conv1.kernel_size[0] * self.conv1.kernel_size[1]))# + 1))
        gc2 = self.gate(s * self.conv2.ec(t))  # .view(self.conv2.out_channels, (
        # self.conv2.in_channels * self.conv2.kernel_size[0] * self.conv2.kernel_size[1]))# + 1))
        gc3 = self.gate(s * self.conv3.ec(t))  # .view(self.conv3.out_channels, (
        # self.conv3.in_channels * self.conv3.kernel_size[0] * self.conv3.kernel_size[1]))# + 1))
        # gc4 = self.gate(s * self.conv4.ec(t))#.view(self.conv4.out_channels, (
        #         #self.conv4.in_channels * self.conv4.kernel_size[0] * self.conv4.kernel_size[1]))# + 1))
        return [gc1, gc2, gc3]  # ,gc4]

    def get_total_mask(self, t, labels, s):
        task = torch.autograd.Variable(torch.LongTensor(labels.data.cpu().numpy()).cuda())
        # t =  torch.autograd.Variable(torch.LongTensor([t]).cuda())
        masks = self.mask(task, None, s=s)
        m0 = torch.max(masks[0], 0)[0].view(1, -1)
        m1 = torch.max(masks[1], 0)[0].view(1, -1)
        m2 = torch.max(masks[2], 0)[0].view(1, -1)
        return [m0, m1, m2]  # + m3

    def get_view_for(self, n, masks):
        gc1, gc2, gc3 = masks
        if n == 'conv1.weight':
            return gc1.data[:, :].contiguous().view(self.conv1.weight.shape)
        elif n == 'conv1.bias':
            return gc1.data[:, 0].contiguous().view(-1)

        elif n == 'conv2.weight':
            return gc2.data[:, :].contiguous().view(self.conv2.weight.shape)
        elif n == 'conv2.bias':
            return gc2.data[:, 0].contiguous().view(-1)

        elif n == 'conv3.weight':
            return gc3.data[:, :].contiguous().view(self.conv3.weight.shape)
        elif n == 'conv3.bias':
            return gc3.data[:, 0].contiguous().view(-1)
        return None


class netD(nn.Module):
    def __init__(self, ndf, nc, out_class=1):
        super(netD, self).__init__()
        self.nc = nc
        self.momentum = 0.1
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(nc, ndf, 5, 2, 2, bias=False)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 5, 2, 2, bias=False)
        self.BatchNorm2 = BatchNorm2d(ndf * 2, momentum=self.momentum, track_running_stats=True)
        # self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 5, 2, 2, bias=False)

        self.BatchNorm3 = BatchNorm2d(ndf * 4, momentum=self.momentum, track_running_stats=True)
        self.output_size = ndf * 4 * 4 * 4
        self.disc_linear = nn.Linear(self.output_size, 1)  # .append(nn.Linear(ndf, 1))
        self.aux_linear = nn.Linear(self.output_size, out_class)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.ndf = ndf
        self.apply(weights_init)

    def forward(self, input):
        batch_size = input.size()[0]
        x = self.conv1(input)
        x = self.LeakyReLU(x)
        x = self.conv2(x)
        x = self.BatchNorm2(x)
        x = self.LeakyReLU(x)
        x = self.conv3(x)
        x = self.BatchNorm3(x)
        x = self.LeakyReLU(x)
        x = x.view(batch_size, -1)
        c = self.aux_linear(x)
        s = self.disc_linear(x)
        return s.view(-1), c
