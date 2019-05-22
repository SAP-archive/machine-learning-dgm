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
import torch.nn as nn
import torch
import numpy as np
from copy import copy
import torch.nn.functional as F
from networks.resnet import resnet18
import math
from torch.nn.parameter import Parameter
from utils.utils import weights_init_g


class Linear_extandable(nn.Linear):
    def __init__(self, in_features, out_features, num_tasks=1, bias=True, smax=1000, device=None):
        super(Linear_extandable, self).__init__(in_features, out_features, bias)
        self.in_features = in_features
        self.out_features = out_features
        self.smax = smax
        self.device = device
        # self.reset_parameters()
        self.ec = torch.nn.Embedding(1, (self.in_features) * self.out_features)
        self.ec_b = None
        if self.bias is not None:
            self.ec_b = torch.nn.Embedding(1, self.out_features)

        self.s = torch.nn.ParameterList()
        self.ec.weight.data.fill_(0)#6/self.smax)
        self.ec_past = torch.sparse.FloatTensor(10, self.out_features, self.in_features)

        self.prev_weight_shape = self.weight.shape

    def forward(self, inputx, mask, d_in, d_out, output_size=None):
        if mask is not None:
            bias = None
            if self.bias is not None:
                bias = self.bias[:d_out] * mask[1][:d_out]
            out = F.linear(inputx, self.weight[:d_out, :d_in] * mask[0][:d_out, :d_in], bias)
        else:
            out = F.linear(inputx, self.weight[:d_out, :d_in], self.bias[:d_out])

        return out

    def extand(self, delta_in_features, delta_out_features):
        w_old = self.weight.data.clone()
        b_old = None
        if self.bias is not None:
            b_old = self.bias.data.clone()
        self.out_features += delta_out_features[0]
        self.out_features = int(16 * math.ceil(self.out_features / 16.))
        self.in_features += delta_in_features

        self.weight = Parameter(torch.Tensor(self.out_features, self.in_features).cuda(self.device))

        if self.bias is not None:
            self.bias = Parameter(torch.Tensor(self.out_features).cuda(self.device))

        self.apply(weights_init_g)
        self.weight.data[:w_old.shape[0], :w_old.shape[1]].copy_(w_old)
        if self.bias is not None:
            self.bias.data[:b_old.shape[0]].copy_(b_old)

        del (w_old)
        del (b_old)
        torch.cuda.empty_cache()
        return self.weight.shape

    def expand_embeddings(self, n_new_classes, t, mask):
        # extand amd store the masks
        a = self.ec_past.to_dense().cpu()
        a[t] = mask[0]
        self.ec_past = torch.sparse.FloatTensor((a == 1).nonzero().t(),
                                                torch.ones((a == 1).nonzero().shape[0]),
                                                torch.Size([10, self.weight.shape[0], self.weight.shape[1]])).cuda(self.device)
        del (a)
        torch.cuda.empty_cache()
        self.ec = torch.nn.Embedding(1, self.weight.shape[0] * self.weight.shape[1]).cuda(self.device)
        self.ec.weight.data.fill_(6/self.smax)
        self.prev_weight_shape = self.weight.shape

class Plastic_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_tasks=1, smax=1000, device=None):
        super(Plastic_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                             bias)

        self.ec_bias = None
        self.device = device
        if bias:
            self.ec_bias = torch.nn.Embedding(num_tasks, out_channels)  # .cuda()
            self.ec_bias.weight.data.fill_(0)

        self.ec = torch.nn.Embedding(1, out_channels * ((in_channels * kernel_size * kernel_size)))  # .cuda()
        self.smax = smax
        self.ec.weight.data.fill_(0) # 6/self.smax)
        self.ec_past = torch.sparse.FloatTensor(10, self.weight.shape[0], self.weight.shape[1], self.weight.shape[2],
                                                self.weight.shape[3])
        self.prev_weight_shape = self.weight.shape

    def forward(self, inputx, mask, d_in, d_out):
        # self.prev_weight_shape=self.weight.shape
        bias = None
        if mask is not None:
            mask_ = mask[0]
            if self.bias is not None:
                mask_b = mask[1]
                bias = self.bias[:d_out] * mask_b[:d_out]
            out = F.conv2d(inputx, self.weight[:d_out, :d_in, :, :] * mask_[:d_out, :d_in, :, :], bias, self.stride,
                           self.padding, groups=self.groups, dilation=self.dilation)

        else:
            if self.bias is not None:
                bias = self.bias[:d_out]
            out = F.conv2d(inputx, self.weight[:d_out, :d_in, :, :], bias, self.stride, self.padding,
                           groups=self.groups, dilation=self.dilation)
        return out

    def expand(self, input_channels, out_channels):
        b_old = None
        w_old = self.weight.data.clone()
        if self.bias is not None:
            b_old = self.bias.data.clone()
        self.out_channels += out_channels
        self.in_channels += input_channels
        self.weight = Parameter(
            torch.Tensor(self.out_channels, (self.in_channels // self.groups), *self.kernel_size).cuda(self.device))
        if self.bias is not None:
            self.bias = Parameter(torch.Tensor(self.out_channels)).cuda(self.device)
        self.apply(weights_init_g)
        self.weight.data[:w_old.shape[0]:, :w_old.shape[1], :, :].copy_(w_old)
        if self.bias is not None:
            self.bias.data[:b_old.shape[0]].copy_(b_old)
        del(w_old)
        torch.cuda.empty_cache()


        return self.weight.shape

    def expand_embeddings(self, n_new_classes, t, mask):
        a = self.ec_past.to_dense().cpu()
        a[t] = mask[0]
        if (a == 1).nonzero().shape[0] == 0:
            self.ec_past = torch.sparse.FloatTensor(10, self.weight.shape[0], self.weight.shape[1],
                                                    self.weight.shape[2], self.weight.shape[3]).cuda(self.device)
        else:
            self.ec_past = torch.sparse.FloatTensor((a == 1).nonzero().t(),
                                                    torch.ones((a == 1).nonzero().shape[0]),
                                                    torch.Size([10, self.weight.shape[0], self.weight.shape[1],
                                                                self.weight.shape[2], self.weight.shape[3]])).cuda(
                self.device)
        del (a)
        torch.cuda.empty_cache()
        self.ec = torch.nn.Embedding(1, self.out_channels * (
        (self.in_channels * self.kernel_size[0] * self.kernel_size[0]))).cuda(self.device)
        self.ec.weight.data.fill_(6/self.smax)
        self.prev_weight_shape = self.weight.shape

def avg_pool2d(x):
    '''Twice differentiable implementation of 2x2 average pooling.'''
    return (x[:, :, ::2, ::2] + x[:, :, 1::2, ::2] + x[:, :, ::2, 1::2] + x[:, :, 1::2, 1::2]) / 4

class GeneratorBlock(nn.Module):
    '''ResNet-style block for the generator model.'''
    def __init__(self, in_chans, out_chans, smax, upsample=False, device="cuda"):
        super(GeneratorBlock, self).__init__()

        self.gate = torch.nn.Sigmoid()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.device = device

        self.upsample = upsample
        self.shortcut_conv = Plastic_Conv2d(self.in_chans, self.out_chans, kernel_size=1, bias=False,
                                            device=self.device)
        if self.in_chans != self.out_chans:
            self.cap_shortcut = [self.out_chans]
        else:
            self.cap_shortcut = [None]
        self.BatchNorm1s = torch.nn.ModuleList()
        self.BatchNorm1s.append(torch.nn.BatchNorm2d(in_chans))
        self.conv1 = Plastic_Conv2d(in_chans, in_chans, kernel_size=3, padding=1, bias=False, device=self.device)
        self.cap_conv1 = [in_chans]
        self.BatchNorm2s = torch.nn.ModuleList()
        self.BatchNorm2s.append(torch.nn.BatchNorm2d(in_chans))
        self.conv2 = Plastic_Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False, device=self.device)
        self.cap_conv2 = [out_chans]

    def extand(self, t, inp_dim, c1, c2, c_s, smax):
        print("inp_dim", inp_dim)

        n_out_conv1 = self.conv1.weight.data.shape[0]
        n_params_conv1 = inp_dim * n_out_conv1 * self.conv1.kernel_size[0] * self.conv1.kernel_size[1] # n_params after adding input channels
        n_add_out_conv1 = max(math.ceil((c1[0] - (n_params_conv1 - np.prod(self.conv1.weight.size()).item())) /
            (self.conv1.weight.data.shape[1] * self.conv1.kernel_size[0] * self.conv1.kernel_size[1])), 0)
        delta_in_conv_1 = inp_dim - self.in_chans

        ws_0 = self.conv1.expand(delta_in_conv_1, n_add_out_conv1)

        n_in_conv2 = self.conv1.weight.shape[0]
        n_out_conv2 = self.conv2.weight.data.shape[0]
        n_params_conv2 = n_in_conv2 * n_out_conv2 * self.conv2.kernel_size[0] * self.conv2.kernel_size[1]  # n_params after adding input channels
        n_add_out_conv2 = max(math.ceil((c2[0] - (n_params_conv2 - np.prod(self.conv2.weight.size()).item())) /
                                        (self.conv2.weight.data.shape[1] * self.conv2.kernel_size[0] *
                                         self.conv2.kernel_size[1])), 0)

        ws_1 = self.conv2.expand(n_add_out_conv1, n_add_out_conv2)
        self.BatchNorm1s.append(torch.nn.BatchNorm2d(self.conv1.in_channels).cuda(self.device))
        delta_conv2 = 0
        self.in_chans += delta_in_conv_1
        self.out_chans += n_add_out_conv2
        ws_s = self.shortcut_conv.weight.shape

        if self.in_chans != self.out_chans:
            n_in_conv_sc = inp_dim
            n_out_conv_sc = self.shortcut_conv.weight.data.shape[0]
            n_params_conv_sc = n_in_conv_sc * n_out_conv_sc * self.shortcut_conv.weight.shape[0] * self.shortcut_conv.weight.shape[1]  # n_params after adding input channels
            n_add_out_conv_sc = max(math.ceil((c_s[0] - (n_params_conv_sc - np.prod(self.shortcut_conv.weight.size()).item())) /
                                            (self.shortcut_conv.weight.data.shape[1] * self.shortcut_conv.kernel_size[0] *
                                             self.shortcut_conv.kernel_size[1])), 0)
            ws_s = self.shortcut_conv.expand((self.conv1.weight.shape[1] - self.shortcut_conv.weight.shape[1]),   n_add_out_conv_sc)

            if self.shortcut_conv.weight.shape[0] > self.conv2.weight.shape[0]:
                delta_conv2 = self.shortcut_conv.weight.shape[0] - self.conv2.weight.shape[0]
                _ = self.conv2.expand(0, delta_conv2)

            else:
                self.shortcut_conv.expand(0, self.conv2.weight.shape[0] - self.shortcut_conv.weight.shape[0])
            self.cap_shortcut.append(self.shortcut_conv.out_channels)

        else:
            self.cap_shortcut.append(None)

        self.out_chans += delta_conv2
        self.BatchNorm2s.append(torch.nn.BatchNorm2d(self.conv2.in_channels).cuda(self.device))
        self.cap_conv1.append(self.conv1.weight.shape[0])
        self.cap_conv2.append(self.conv2.weight.shape[0])
        torch.cuda.empty_cache()
        return ws_1[0] + delta_conv2

    def expand_embeddings(self, n_new_classes, t, mask):
        self.conv1.expand_embeddings(n_new_classes, t, mask[0])
        self.conv2.expand_embeddings(n_new_classes, t, mask[1])
        self.shortcut_conv.expand_embeddings(n_new_classes, t, mask[2])

        prev_weight_shape_0 = copy(self.conv1.prev_weight_shape)
        prev_weight_shape_1 = copy(self.conv2.prev_weight_shape)
        prev_weight_shape_s = copy(self.shortcut_conv.prev_weight_shape)
        self.conv1.prev_weight_shape = self.conv1.weight.shape
        self.conv2.prev_weight_shape = self.conv2.weight.shape
        self.shortcut_conv.prev_weight_shape = self.shortcut_conv.weight.shape
        return [prev_weight_shape_0, prev_weight_shape_1, prev_weight_shape_s]  # , prev_weight_shape_2]#, prev_weight_shape_3]

    def forward(self, input_, cap_prev, task, s=1, past_generation=False):
        x = input_
        t = task
        if not past_generation:
            masks = self.mask(task, s=s)
        else:
            masks = self.eval_masks(task)
        gc1, gc2, gc_s = masks
        if self.upsample:
            shortcut = nn.functional.upsample(x, scale_factor=2, mode='nearest')
        else:
            shortcut = x
        if self.cap_shortcut[t] is not None:
            shortcut = self.shortcut_conv(shortcut, gc_s, cap_prev, self.cap_shortcut[t])
        x = self.BatchNorm1s[t](x)  # (x, cap_prev)
        x = nn.functional.relu(x, inplace=True)
        if self.upsample:
            x = nn.functional.upsample(x, scale_factor=2, mode='nearest')
        x = self.conv1(x, gc1, cap_prev, self.cap_conv1[t])
        x = self.BatchNorm2s[t](x)  # (x,self.cap_conv1[t])
        x = nn.functional.relu(x, inplace=True)
        x = self.conv2(x, gc2, self.cap_conv1[t], self.cap_conv2[t])

        return x + shortcut, self.cap_conv2, masks

    def eval_masks(self, task):
        gc1 = self.conv1.ec_past.to_dense()[task].squeeze(0)
        gc2 = self.conv2.ec_past.to_dense()[task].squeeze(0)
        gcs = self.shortcut_conv.ec_past.to_dense()[task].squeeze(0)
        return [[gc1, None], [gc2, None], [gcs, None]]

    def mask(self, t, s=1, test=False):
        t = torch.autograd.Variable(torch.LongTensor([0]).cuda(self.device))
        gc1 = self.gate(s * self.conv1.ec(t)).contiguous().view(self.conv1.weight.size())
        gc1_b = None
        if self.conv1.ec_bias is not None:
            gc1_b = self.gate(s * self.conv1.ec_bias(t)).view(-1)

        gc2 = self.gate(s * self.conv2.ec(t)).view(self.conv2.weight.size())
        gc2_b = None
        if self.conv2.ec_bias is not None:
            gc2_b = self.gate(s * self.conv2.ec_bias(t)).view(-1)

        gc_s = self.gate(s * self.shortcut_conv.ec(t)).view(self.shortcut_conv.weight.size())
        gcs_b = None
        if self.shortcut_conv.ec_bias is not None:
            gcs_b = self.gate(s * self.shortcut_conv.ec_bias(t)).view(-1)
        return [[gc1, gc1_b], [gc2, gc2_b], [gc_s, gc2_b]]

    def get_view_for(self, n, masks):
        gc1, gc2, gc_s = masks
        if n.endswith('conv1.weight'):
            return gc1[0][:, :].contiguous().view(self.conv1.weight.shape)
        elif n.endswith('conv1.bias'):
            gc1 = gc1[1][:, :].contiguous().view(self.conv1.bias.shape)
            return gc1.data.view(-1)
        elif n.endswith('conv2.weight'):
            return gc2[0][:, :].contiguous().view(self.conv2.weight.shape)
        elif n.endswith('conv2.bias'):
            gc2 = gc2[1][:, :].contiguous().view(self.conv2.bias.shape)
            return gc2.data.view(-1)
        elif n.endswith('shortcut_conv.weight'):
            return gc_s[0][:, :].contiguous().view(self.shortcut_conv.weight.shape)
        elif n.endswith('shortcut_conv.bias'):
            gc_s = gc_s[1][:, :].contiguous().view(self.shortcut_conv.bias.shape)
            return gc_s.data.view(-1)
        return None


class netG(nn.Module):
    def __init__(self, nz, ngf, nc, smax, device):
        super(netG, self).__init__()
        self.nz = nz
        self.gate = torch.nn.Sigmoid()
        self.nc = nc
        self.ngf = ngf
        self.device = device
        self.tanh = nn.Tanh()
        self.smax = smax
        self.scalor = 1
        self.feats = ngf
        self.fc1 = Linear_extandable(self.nz, 4 * 4 * self.feats, bias=False, device=self.device)
        self.cap_fc0 = [ 4 * 4 *  self.feats]
        self.shape_fc_1_out = [4 * 4 * self.feats]
        self.block1 = GeneratorBlock(self.feats, self.feats, smax, upsample=True,
                                     device=self.device)
        self.block2 = GeneratorBlock(self.feats, self.feats, smax, upsample=True,
                                     device=self.device)
        self.block3 = GeneratorBlock(self.feats, self.feats, smax, upsample=True,
                                     device=self.device)
        self.output_bns = torch.nn.ModuleList()
        self.output_bns.append(torch.nn.BatchNorm2d(self.scalor * self.feats))
        self.apply(weights_init_g)
        self.last = torch.nn.ModuleList()
        self.efc1 = torch.nn.Embedding(10, self.scalor * 4 * 4 * self.feats)
        self.efc1.weight.data.fill_( 6/smax)

    def extand(self, t, extention, smax):
        print(extention)
        ws_fc = self.fc1.extand(0, [math.ceil(extention[0][0] / (self.nz)), extention[0][1]])
        self.cap_fc0.append(self.fc1.weight.shape[0])
        a = int(math.ceil(self.fc1.weight.shape[0] / 16.)) # desired output n_channels x 4 x 4 -> output of fc1 should be devidable by 16
        #print("a", a - self.block1.BatchNorm1s[t].weight.shape[0])
        out_dim = self.block1.extand(t, a, extention[1], extention[2], extention[3], smax)
        out_dim = self.block2.extand(t, out_dim, extention[4],
                                     extention[5], extention[6], smax)
        out_dim = self.block3.extand(t, out_dim, extention[7],
                                     extention[8], extention[9], smax)
        # self.output_bn.extand(extention[6])
        self.output_bns.append(torch.nn.BatchNorm2d(out_dim).cuda(self.device))
        ws_0 = [self.block1.conv1.weight.shape, self.block1.conv2.weight.shape, self.block1.shortcut_conv.weight.shape]
        ws_1 = [self.block2.conv1.weight.shape, self.block2.conv2.weight.shape, self.block2.shortcut_conv.weight.shape]
        ws_2 = [self.block3.conv1.weight.shape, self.block3.conv2.weight.shape, self.block3.shortcut_conv.weight.shape]
        return [ws_fc] + ws_0 + ws_1 + ws_2

    def total_size_n_params(self):
        size = 0
        size += np.prod(self.efc1.weight.size())
        size += np.prod(self.block1.conv1.weight.size()) + np.prod(self.block1.conv2.weight.size()) + \
                np.prod(self.block1.shortcut_conv.weight.size())
        size += np.prod(self.block2.conv1.weight.size()) + np.prod(self.block2.conv2.weight.size()) + \
                np.prod(self.block2.shortcut_conv.weight.size())
        size += np.prod(self.block3.conv1.weight.size()) + np.prod(self.block3.conv2.weight.size()) + \
                np.prod(self.block3.shortcut_conv.weight.size())
        return size

    def total_size(self):
        size = 0
        size += self.efc1.weight.shape[0]
        size += self.block1.conv1.weight.shape[0] + self.block1.conv2.weight.shape[0] + \
                self.block1.shortcut_conv.weight.shape[0]
        size += self.block2.conv1.weight.shape[0] + self.block2.conv2.weight.shape[0] + \
                self.block2.shortcut_conv.weight.shape[0]
        size += self.block3.conv1.weight.shape[0] + self.block3.conv2.weight.shape[0] + \
                self.block3.shortcut_conv.weight.shape[0]
        return size


    def expand_embeddings(self, n_new_classes, t, mask):
        self.fc1.expand_embeddings(n_new_classes, t, mask[0])
        prev_weight_shape_fc = [copy(self.fc1.prev_weight_shape)]
        self.fc1.prev_weight_shape = self.fc1.weight.shape

        # self.fc1.expand_embeddings(n_new_classes)
        prev_weight_shape_0 = self.block1.expand_embeddings(n_new_classes, t, mask[1:4])
        # self.block1.expand_embeddings(n_new_classes)
        prev_weight_shape_1 = self.block2.expand_embeddings(n_new_classes, t, mask[4:7])
        # self.block2.expand_embeddings(n_new_classes)
        prev_weight_shape_2 = self.block3.expand_embeddings(n_new_classes, t, mask[7:10])
        return [prev_weight_shape_fc, prev_weight_shape_0, prev_weight_shape_1, prev_weight_shape_2]

    def forward(self, input_, task=0, s=1, past_generation=False, lables=None):
        # task =  torch.autograd.Variable(torch.LongTensor([t]).cuda())
        if not past_generation:
            gfc1 = self.mask(task, s=s)
        else:
            gfc1 = self.eval_masks(task)
        t = task
        x = input_
        x = self.fc1(x.view(-1, self.nz), gfc1, self.nz, self.cap_fc0[t])
        x = x.view(-1, int(x.shape[1] / 16.), 4, 4)
        x, prev_cap, masks1 = self.block1(x, x.shape[1], t, s, past_generation=past_generation)
        x, prev_cap, masks2 = self.block2(x, prev_cap[t], t, s, past_generation=past_generation)
        x, prev_cap, masks3 = self.block3(x, prev_cap[t], t, s, past_generation=past_generation)
        x = self.output_bns[t](x)
        x = nn.functional.relu(x, inplace=False)
        output = torch.stack(
            [self.tanh(self.last[c](x[i].unsqueeze(0), None, prev_cap[t], self.nc)) for i, c in
             enumerate(lables)]).squeeze(1)

        masks = [gfc1] + masks1 + masks2 + masks3 #+ masks4

        return output, masks

    def eval_masks(self, task):
        gc1 = self.fc1.ec_past.to_dense()[task].squeeze(0)
        return [gc1, None]

    def mask(self, t, s=1, test=False):
        t = torch.autograd.Variable(torch.LongTensor([0]).cuda(self.device))
        gf0 = self.gate(s * self.fc1.ec(t)).view(self.fc1.weight.shape)
        gf0_b = None
        if self.fc1.ec_b is not None:
            gf0_b = self.gate(s * self.fc1.ec_b(t)).view(-1)
        return [gf0, gf0_b]

    def get_total_mask(self, t, s):
        m0 = self.mask(t, s)
        m1 = self.block1.mask(t, s)
        m2 = self.block2.mask(t, s)
        m3 = self.block3.mask(t, s)

        return [m0] + m1 + m2 + m3

    def get_total_mask_eval(self, t):
        m0 = self.eval_masks(t)
        m1 = self.block1.eval_masks(t)
        m2 = self.block2.eval_masks(t)
        m3 = self.block3.eval_masks(t)

        return [m0] + m1 + m2 + m3

    def get_view_for(self, n, masks):
        gfc1 = masks
        if n == 'fc1.weight':
            return gfc1[0].expand_as(self.fc1.weight)
        elif n == 'fc1.bias':
            return gfc1[1].data[:, :].view(-1)

        return None

class DiscriminatorBlock(nn.Module):
    '''ResNet-style block for the discriminator model.'''
    def __init__(self, in_chans, out_chans, downsample=False, first=False):
        super(DiscriminatorBlock, self).__init__()

        self.downsample = downsample
        self.first = first

        if in_chans != out_chans:
            self.shortcut_conv = nn.Conv2d(in_chans, out_chans, kernel_size=1)
        else:
            self.shortcut_conv = None
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1)

    def forward(self, *inputs):
        x = inputs[0]

        if self.downsample:
            shortcut = avg_pool2d(x)
        else:
            shortcut = x

        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)

        if not self.first:
            x = nn.functional.relu(x, inplace=False)
        x = self.conv1(x)
        x = nn.functional.relu(x, inplace=False)
        s = self.bn(x)
        x = self.conv2(x)
        if self.downsample:
            x = avg_pool2d(x)

        return x + shortcut

class netD(nn.Module):
    def __init__(self, feature_size, n_classes, device):
        # Network architecture
        super(netD, self).__init__()
        self.device = device
        self.feats = feature_size
        self.feature_extractor = resnet18()
        self.feature_extractor.fc = \
            nn.Linear(self.feature_extractor.fc.in_features, feature_size)
        self.bn = nn.BatchNorm1d(feature_size, momentum=0.01)
        self.ReLU = nn.ReLU()
        self.aux_linear = nn.Linear(feature_size, n_classes, bias=False)
        self.disc_linear = nn.Linear(feature_size, 1)
        self.n_classes = n_classes
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.bn(x)
        x = self.ReLU(x)
        c = self.aux_linear(x)
        s = self.disc_linear(x)
        return s.view(-1), c
