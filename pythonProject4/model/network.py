import torch
import torch.nn as nn
import math


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


def conv3x3(in_planes, out_planes):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)


def conv3x3_down(in_planes, out_planes):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False)


import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class ChannelGate(nn.Module):  #先将输入avgpool&maxpool后过到MLP感知机里 相加而后用sigmoid后 和原来x相乘 （另一种注意力）
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        # shared MLP 多层感知机
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale
def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial2 = BasicConv(2, 2, 7, stride=1, padding=(3 - 1) // 2, relu=True)
        self.downsample1 = nn.Conv2d(2, 2, kernel_size, stride=2, padding=(kernel_size - 1) // 2)
        self.upsample1 = nn.ConvTranspose2d(2, 2, kernel_size + 1, stride=2, padding=(kernel_size - 1) // 2)
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_down = self.downsample1(x_compress)
        x_down = self.compress(x_down)
        x_down = self.spatial2(x_down)
        x_down = F.interpolate(x_down,
                               size=[x.size(2), x.size(3)],
                               mode='bilinear', align_corners=False)
        x_out = x_down * 0.1 + x_compress
        x_out = self.spatial(x_out)
        scale = F.sigmoid(x_out)
        return x * scale
class ATTEN(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(ATTEN, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)

        x_out = self.SpatialGate(x_out)
        return x_out


class RDB(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate, scale = 1.0):
    super(RDB, self).__init__()
    nChannels_ = nChannels
    self.scale = scale
    modules = []
    for i in range(nDenselayer):
        modules.append(make_dense(nChannels_, growthRate))
        nChannels_ += growthRate
    self.dense_layers = nn.Sequential(*modules)
    self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out) * self.scale
    out = out + x
    return out

class make_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out



class CFMSAN(nn.Module):
    def __init__(self):
        super(CFMSAN, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv7= nn.Conv2d(384, 128, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn1 = nn.BatchNorm2d(128)
        self.downsample1=nn.Conv2d(128,128,kernel_size=3,stride=2,padding=1,bias=False)
        self.upsample=nn.ConvTranspose2d(128,128,kernel_size=4,stride=2,padding=1)
        #self.upsample = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, )
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.sigmond1 = nn.Sigmoid()
        self.cbamlayer = ATTEN(128)
        self.cbamlayer1 = ATTEN(384)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):  # 3x320x320
        x = self.relu(self.conv0(x))  # ->64x320x320
        x = self.relu(self.conv1(x))  # ->128x320x320
        x3_3=self.relu(self.conv2(x))
        x5_5=self.relu(self.conv3(x))
        x7_7=self.relu(self.conv4(x))
        xfinal=torch.concat((x3_3,x5_5,x7_7),dim=1)
        x_CBAM = self.cbamlayer1(xfinal)
        x_CBAM=self.relu(self.conv7(x_CBAM))
        x_down=self.downsample1(x)
        x_down_3_3=self.relu(self.conv2(x_down))
        x_down_5_5 = self.relu(self.conv2(x_down))
        x_down_7_7 = self.relu(self.conv2(x_down))
        x_down_final = torch.concat((x_down_3_3, x_down_5_5, x_down_7_7 ), dim=1)
        x_down_final=   self.cbamlayer1(x_down_final)
        x_down_final = self.relu(self.conv7(x_down_final))
        x_up=self.upsample(x_down_final)
        x_out=x_up+x_CBAM
        x_out=self.relu(self.conv6(x_out))
        x_out=self.sigmond1(x_out)

        return x_out


def epsanet50():
    model = CFMSAN()
    return model


