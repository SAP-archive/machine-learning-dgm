import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from utils.utils import weights_init_g, weights_init


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


class ConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 groups=1, bias=True,
                 num_tasks=1, out_size=None, batch_size=64):
        super(ConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding,
                                              groups, bias, dilation)

    def forward(self, inputx, d_in, d_out, output_size=None):
        output_padding = self._output_padding(inputx, output_size, self.stride, self.padding, self.kernel_size)
        out = F.conv_transpose2d(inputx, self.weight[:d_in, :d_out, :, :], self.bias, self.stride, self.padding,
                                 output_padding, groups=self.groups, dilation=self.dilation)
        return out

    def extand(self, input_channels, out_channels):
        w_old = self.weight.data.clone()

        self.out_channels += out_channels
        self.in_channels += input_channels
        self.weight = Parameter(
            torch.Tensor(self.in_channels, self.out_channels // self.groups, *self.kernel_size).cuda())
        if self.bias is not None:
            b_old = self.bias.data.clone()
            self.bias = Parameter(torch.Tensor(self.in_channels).cuda())
        self.apply(weights_init_g)
        self.weight.data[:w_old.shape[0]:, :w_old.shape[1], :, :].copy_(w_old)
        if self.bias is not None:
            self.bias.data[:b_old.shape[0]].copy_(b_old)


class BatchNorm2d_plastic(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm2d_plastic, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, inputx, out_dim):
        self._check_input_dim(inputx)
        out = F.batch_norm(inputx, self.running_mean[:out_dim], self.running_var[:out_dim], self.weight[:out_dim],
                           self.bias[:out_dim],
                           self.training or not self.track_running_stats, self.momentum, self.eps)
        return out


class netG(nn.Module):
    def __init__(self, nz, ngf, nc, smax, scalor=1, n_classes=1):
        super(netG, self).__init__()

        self.nz = nz
        self.gate = torch.nn.Sigmoid()

        self.nc = nc
        self.ngf = ngf
        self.scalor = scalor
        self.smax = smax
        self.ReLU = nn.ReLU(True)
        self.Tanh = nn.Tanh()
        self.conv1 = ConvTranspose2d(nz, self.scalor * ngf * 4, 4, 1, 0, bias=False)
        self.cap_conv1 = [self.scalor * ngf * 4]
        self.BatchNorms0 = torch.nn.ModuleList()
        self.BatchNorms0.append(torch.nn.BatchNorm2d(self.scalor * ngf * 4).apply(weights_init_g))

        self.conv2 = ConvTranspose2d(self.scalor * ngf * 4, self.scalor * ngf * 2, 4, 2, 1, bias=False)
        self.cap_conv2 = [self.scalor * ngf * 2]
        self.BatchNorms1 = torch.nn.ModuleList()
        self.BatchNorms1.append(torch.nn.BatchNorm2d(self.scalor * ngf * 2).apply(weights_init_g))

        self.conv3 = ConvTranspose2d(self.scalor * ngf * 2, self.scalor * ngf * 1, 4, 2, 1, bias=False)
        self.cap_conv3 = [self.scalor * ngf * 1]
        self.BatchNorms2 = torch.nn.ModuleList()
        self.BatchNorms2.append(torch.nn.BatchNorm2d(self.scalor * ngf * 1).apply(weights_init_g))

        self.apply(weights_init)

        self.last = self.last = torch.nn.ModuleList()
        self.ec1 = torch.nn.Embedding(10, ngf * 4 * self.scalor)
        self.ec2 = torch.nn.Embedding(10, ngf * 2 * self.scalor)
        self.ec3 = torch.nn.Embedding(10, ngf * 1 * self.scalor)
        # self.ec4=torch.nn.Embedding(10,ngf * 1*self.scalor)

        self.ec1.weight.data.fill_(0)
        self.ec2.weight.data.fill_(0)
        self.ec3.weight.data.fill_(0)

    def forward(self, input, t, lables=None, s=1, t_mix=None):
        task = torch.autograd.Variable(torch.LongTensor([t]).cuda())
        masks = self.mask(task, s=s)

        gc1, gc2, gc3 = masks
        x = self.conv1(input.view(-1, self.nz, 1, 1), self.nz, self.cap_conv1[t])
        x = x * gc1[:, :self.cap_conv1[t]].view(1, -1, 1, 1).expand_as(x)
        x = self.BatchNorms0[t](x)  # , self.cap_conv1[t])
        x = self.ReLU(x)

        x = self.conv2(x, self.cap_conv1[t], self.cap_conv2[t])
        x = x * gc2[:, :self.cap_conv2[t]].view(1, -1, 1, 1).expand_as(x)
        x = self.BatchNorms1[t](x)  # , self.cap_conv2[t])
        x = self.ReLU(x)

        x = self.conv3(x, self.cap_conv2[t], self.cap_conv3[t])
        x = x * gc3[:, :self.cap_conv3[t]].view(1, -1, 1, 1).expand_as(x)
        x = self.BatchNorms2[t](x)  # , self.cap_conv3[t])
        x = self.ReLU(x)
        output = self.Tanh(self.last[t](x, self.cap_conv3[t], self.nc))

        return output, masks

    def extand(self, t, extention, smax):
        print(extention)

        # self.BatchNorm0.extand(a)
        self.conv1.extand(0, extention[0])
        self.BatchNorms0.append(torch.nn.BatchNorm2d(extention[0] + self.BatchNorms0[t].weight.shape[0]).cuda())

        self.conv2.extand(extention[0], extention[1])
        self.BatchNorms1.append(torch.nn.BatchNorm2d(extention[1] + self.BatchNorms1[t].weight.shape[0]).cuda())

        self.conv3.extand(extention[1], extention[2])
        self.BatchNorms2.append(torch.nn.BatchNorm2d(extention[2] + self.BatchNorms2[t].weight.shape[0]).cuda())

        self.cap_conv1.append(self.conv1.weight.shape[1])
        self.cap_conv2.append(self.conv2.weight.shape[1])
        self.cap_conv3.append(self.conv3.weight.shape[1])

        ec_1 = self.ec1.weight.data.clone()
        ec_2 = self.ec2.weight.data.clone()
        ec_3 = self.ec3.weight.data.clone()
        self.ec1 = torch.nn.Embedding(10, self.conv1.weight.shape[1]).cuda()
        self.ec2 = torch.nn.Embedding(10, self.conv2.weight.shape[1]).cuda()
        self.ec3 = torch.nn.Embedding(10, self.conv3.weight.shape[1]).cuda()
        self.ec1.weight.data.fill_(0)
        self.ec2.weight.data.fill_(0)
        self.ec3.weight.data.fill_(0)

        self.ec1.weight.data[:t + 1, :].fill_(-90)
        self.ec2.weight.data[:t + 1, :].fill_(-90)
        self.ec3.weight.data[:t + 1, :].fill_(-90)

        self.ec1.weight.data[:t + 1, :ec_1.shape[1]].copy_(ec_1[:t + 1, :])
        self.ec2.weight.data[:t + 1, :ec_2.shape[1]].copy_(ec_2[:t + 1, :])
        self.ec3.weight.data[:t + 1, :ec_3.shape[1]].copy_(ec_3[:t + 1, :])

        return [self.conv1.weight.shape, self.conv2.weight.shape, self.conv3.weight.shape]

    def mask(self, t, s=1, test=False):
        gc1 = self.gate(s * self.ec1(t))
        gc2 = self.gate(s * self.ec2(t))
        gc3 = self.gate(s * self.ec3(t))
        return [gc1, gc2, gc3]

    def get_view_for(self, n, masks):
        gc1, gc2, gc3 = masks
        if n == 'conv1.weight':
            return gc1.data.view(1, -1, 1, 1).expand_as(self.conv1.weight)
        elif n == 'conv1.bias':
            return gc1.data.view(-1)
        elif n == 'conv2.weight':
            post = gc2.data.view(1, -1, 1, 1).expand_as(self.conv2.weight)
            pre = gc1.data.view(-1, 1, 1, 1).expand_as(self.conv2.weight)
            return torch.min(post, pre)
        elif n == 'conv2.bias':
            return gc2.data.view(-1)
        elif n == 'conv3.weight':
            post = gc3.data.view(1, -1, 1, 1).expand_as(self.conv3.weight)
            pre = gc2.data.view(-1, 1, 1, 1).expand_as(self.conv3.weight)
            return torch.min(post, pre)
        elif n == 'conv3.bias':
            return gc3.data.view(-1)
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