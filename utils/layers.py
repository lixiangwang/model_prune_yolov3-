import torch.nn.functional as F

from utils.utils import *


def make_divisible(v, divisor):
    # Function ensures all layers have a channel number that is divisible by 8
    # https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    return math.ceil(v / divisor) * divisor


class Flatten(nn.Module):
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    def forward(self, x):
        return x.view(x.size(0), -1)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class FeatureConcat(nn.Module):
    def __init__(self, layers):
        super(FeatureConcat, self).__init__()
        self.layers = layers  # layer indices
        self.multiple = len(layers) > 1  # multiple layers flag

    def forward(self, x, outputs):
        return torch.cat([outputs[i] for i in self.layers], 1) if self.multiple else outputs[self.layers[0]]

class element_wise(nn.Module):
    def __init__(self, layers):
        super(element_wise, self).__init__()
        self.layers = layers  # layer indices
        # self.multiple = len(layers) > 1  # multiple layers flag

    def forward(self, x, outputs):
        b,c=outputs[self.layers[0]].size()
        return outputs[self.layers[0]].view(b,c,1,1)*outputs[self.layers[1]]


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class shuffle_1(nn.Module):
    def __init__(self, layers,channel_32,channel_16,channel_8,channel_4):
        super(shuffle_1, self).__init__()
        self.layers = layers  # layer indices
        self.channel_32=channel_32
        self.channel_16=channel_16
        self.channel_8=channel_8
        self.channel_4=channel_4
        # self.multiple = len(layers) > 1  # multiple layers flag
        # self.conv_4 = nn.Conv2d(in_channels=int(channel_4/4), out_channels=int(channel_4/4), kernel_size=3, padding=3 // 2)
        self.conv_16 = nn.Conv2d(in_channels=int(channel_16 / 4), out_channels=int(channel_32 / 4), kernel_size=3,stride=2,padding=3 // 2)
        self.conv_8_1 = nn.Conv2d(in_channels=int(channel_8 / 4), out_channels=int(channel_8 / 4),  kernel_size=3,stride=2, padding=3 // 2)
        self.conv_8_2 = nn.Conv2d(in_channels=int(channel_8 / 4), out_channels=int(channel_32 / 4), kernel_size=3,stride=2,padding=3//2)
        self.conv_4_1 = nn.Conv2d(in_channels=int(channel_4 / 4), out_channels=int(channel_4 / 4),  kernel_size=3,stride=2, padding=3 // 2)
        self.conv_4_2 = nn.Conv2d(in_channels=int(channel_4 / 4), out_channels=int(channel_32 / 4), kernel_size=3,stride=2, padding=3 // 2)
    def forward(self, x,outputs):
        y_32=channel_shuffle(outputs[self.layers[0]],4)
        y_16=channel_shuffle(outputs[self.layers[1]],4)
        y_8 = channel_shuffle(outputs[self.layers[2]], 4)
        y_4 = channel_shuffle(outputs[self.layers[3]], 4)
        y_32 = y_32[:,:int(self.channel_32/4)]
        y_16 = y_16[:,:int(self.channel_16/4)]
        y_16 = self.conv_16(y_16)
        y_8 = y_8[:,:int(self.channel_8 / 4)]
        y_8 = self.conv_8_1(y_8)
        y_8 = self.conv_8_2(y_8)
        y_4 = y_4[:,:int(self.channel_4 / 4)]
        y_4 = self.conv_4_1(y_4)
        y_4 = self.conv_4_1(y_4)
        y_4 = self.conv_4_2(y_4)
        return torch.cat((y_32,y_16,y_8,y_4), 1)

class shuffle_2(nn.Module):
    def __init__(self, layers,channel_32,channel_16,channel_8,channel_4):
        super(shuffle_2, self).__init__()
        self.layers = layers  # layer indices
        self.channel_32=channel_32
        self.channel_16=channel_16
        self.channel_8=channel_8
        self.channel_4=channel_4
        # self.multiple = len(layers) > 1  # multiple layers flag
        self.conv_32 = nn.Conv2d(in_channels=int(channel_32/4), out_channels=int(channel_16/4), kernel_size=3, padding=3 // 2)
        self.upsample_32=nn.Upsample(scale_factor=2)
        # self.conv_16 = nn.Conv2d(in_channels=int(channel_16 / 4), out_channels=int(channel_16 / 4), kernel_size=3,stride=2,padding=3 // 2)
        self.conv_8 = nn.Conv2d(in_channels=int(channel_8/4),out_channels=int(channel_16/4),kernel_size=3,stride=2,padding=3//2)
        self.conv_4_1 = nn.Conv2d(in_channels=int(channel_4 / 4), out_channels=int(channel_4 / 4), kernel_size=3, stride=2, padding=3 // 2)
        self.conv_4_2 = nn.Conv2d(in_channels=int(channel_4 / 4), out_channels=int(channel_16 / 4), kernel_size=3,stride=2, padding=3 // 2)
    def forward(self, x,outputs):
        y_32 = channel_shuffle(outputs[self.layers[0]], 4)
        y_16 = channel_shuffle(outputs[self.layers[1]], 4)
        y_8 = channel_shuffle(outputs[self.layers[2]], 4)
        y_4 = channel_shuffle(outputs[self.layers[3]], 4)
        y_32=self.conv_32(y_32[:,int(self.channel_32/4):int(self.channel_32/2)])
        y_32=self.upsample_32(y_32)
        # print(y_32.shape)
        # print(y_32.shape)
        y_16 = y_16[:,int(self.channel_16/4):int(self.channel_16/2)]
        # y_16=self.conv_16(y_16)
        y_8 = y_8[:,int(self.channel_8 / 4):int(self.channel_8 / 2)]
        y_8 = self.conv_8(y_8)
        y_4 = y_4[:,int(self.channel_4 / 4):int(self.channel_4 / 2)]
        y_4 = self.conv_4_1(y_4)
        y_4 = self.conv_4_2(y_4)
        return torch.cat((y_32,y_16,y_8,y_4), 1)

class shuffle_3(nn.Module):
    def __init__(self, layers,channel_32,channel_16,channel_8,channel_4):
        super(shuffle_3, self).__init__()
        self.layers = layers  # layer indices
        self.channel_32=channel_32
        self.channel_16=channel_16
        self.channel_8=channel_8
        self.channel_4=channel_4
        # self.multiple = len(layers) > 1  # multiple layers flag
        self.conv_32 = nn.Conv2d(in_channels=int(channel_32/4), out_channels=int(channel_8/4), kernel_size=3, padding=3 // 2)
        self.upsample_32=nn.Upsample(scale_factor=4)
        self.conv_16 = nn.Conv2d(in_channels=int(channel_16/4), out_channels=int(channel_8/4), kernel_size=3, padding=3 // 2)
        self.upsample_16=nn.Upsample(scale_factor=2)
        # self.conv_16 = nn.Conv2d(in_channels=int(channel_16 / 4), out_channels=int(channel_16 / 4), kernel_size=3,stride=2,padding=3 // 2)
        # self.conv_8=nn.Conv2d(in_channels=int(channel_8/4),out_channels=int(channel_8/4),kernel_size=3,stride=2,padding=3//2)
        self.conv_4 = nn.Conv2d(in_channels=int(channel_4 / 4), out_channels=int(channel_8 / 4), kernel_size=3,
                                stride=2, padding=3 // 2)
    def forward(self, x,outputs):
        y_32 = channel_shuffle(outputs[self.layers[0]], 4)
        y_16 = channel_shuffle(outputs[self.layers[1]], 4)
        y_8 = channel_shuffle(outputs[self.layers[2]], 4)
        y_4 = channel_shuffle(outputs[self.layers[3]], 4)
        y_32=self.conv_32(y_32[:,int(self.channel_32/2):int(self.channel_32*0.75)])
        y_32=self.upsample_32(y_32)
        # print(y_32.shape)
        # print(y_32.shape)
        y_16 = self.conv_16(y_16[:,int(self.channel_16/2):int(self.channel_16*0.75)])
        y_16 = self.upsample_16(y_16)
        y_8 = y_8[:,int(self.channel_8 / 2):int(self.channel_8*0.75)]
        # y_8 = self.conv_8(y_8)
        y_4 = y_4[:,int(self.channel_4 / 2):int(self.channel_4*0.75)]
        y_4= self.conv_4(y_4)
        return torch.cat((y_32,y_16,y_8,y_4), 1)

class shuffle_4(nn.Module):
    def __init__(self, layers,channel_32,channel_16,channel_8,channel_4):
        super(shuffle_4, self).__init__()
        self.layers = layers  # layer indices
        self.channel_32=channel_32
        self.channel_16=channel_16
        self.channel_8=channel_8
        self.channel_4=channel_4
        # self.multiple = len(layers) > 1  # multiple layers flag
        self.conv_32 = nn.Conv2d(in_channels=int(channel_32/4), out_channels=int(channel_4/4), kernel_size=9, padding=9 // 2)
        self.upsample_32=nn.Upsample(scale_factor=8)
        self.conv_16 = nn.Conv2d(in_channels=int(channel_16/4), out_channels=int(channel_4/4), kernel_size=3,padding=3 // 2)
        self.upsample_16=nn.Upsample(scale_factor=4)
        self.conv_8 = nn.Conv2d(in_channels=int(channel_8/4), out_channels=int(channel_4/4), kernel_size=3,padding=3 // 2)
        self.upsample_8=nn.Upsample(scale_factor=2)
        # self.conv_4 = nn.Conv2d(in_channels=int(channel_4 / 4), out_channels=int(channel_4 / 4), kernel_size=3,stride=2, padding=3 // 2)
    def forward(self, x,outputs):
        # y_32=self.conv_32(outputs[self.layers[0]][:,int(self.channel_32*0.75):self.channel_32])
        # y_32=self.upsample_32(y_32)
        # print(y_32.shape)
        # print(y_32.shape)
        y_32 = channel_shuffle(outputs[self.layers[0]], 4)
        y_16 = channel_shuffle(outputs[self.layers[1]], 4)
        y_8 = channel_shuffle(outputs[self.layers[2]], 4)
        y_4 = channel_shuffle(outputs[self.layers[3]], 4)
        y_16 = self.conv_16(y_16[:,int(self.channel_16*0.75):self.channel_16])
        y_16=self.upsample_16(y_16)
        y_8 =self.conv_8(y_8[:,int(self.channel_8*0.75):self.channel_8])
        y_8 = self.upsample_8(y_8)
        y_4 = y_4[:,int(self.channel_4*0.75):self.channel_4]
        # y_4= self.conv_4(y_4)
        return torch.cat((y_16,y_8,y_4), 1)
'''
#No channel shuffle
class shuffle_1(nn.Module):
    def __init__(self, layers,channel_32,channel_16,channel_8,channel_4):
        super(shuffle_1, self).__init__()
        self.layers = layers  # layer indices
        self.channel_32=channel_32
        self.channel_16=channel_16
        self.channel_8=channel_8
        self.channel_4=channel_4
        # self.multiple = len(layers) > 1  # multiple layers flag
        # self.conv_4 = nn.Conv2d(in_channels=int(channel_4/4), out_channels=int(channel_4/4), kernel_size=3, padding=3 // 2)
        self.conv_16 = nn.Conv2d(in_channels=int(channel_16 / 4), out_channels=int(channel_32 / 4), kernel_size=3,stride=2,padding=3 // 2)
        self.conv_8_1 = nn.Conv2d(in_channels=int(channel_8 / 4), out_channels=int(channel_8 / 4),  kernel_size=3,stride=2, padding=3 // 2)
        self.conv_8_2 = nn.Conv2d(in_channels=int(channel_8 / 4), out_channels=int(channel_32 / 4), kernel_size=3,stride=2,padding=3//2)
        self.conv_4_1 = nn.Conv2d(in_channels=int(channel_4 / 4), out_channels=int(channel_4 / 4),  kernel_size=3,stride=2, padding=3 // 2)
        self.conv_4_2 = nn.Conv2d(in_channels=int(channel_4 / 4), out_channels=int(channel_32 / 4), kernel_size=3,stride=2, padding=3 // 2)
    def forward(self, x,outputs):
        y_32=outputs[self.layers[0]][:,:int(self.channel_32/4)]
        # print(y_32.shape)
        # print(y_32.shape)
        y_16 = outputs[self.layers[1]][:,:int(self.channel_16/4)]
        y_16=self.conv_16(y_16)
        y_8 = outputs[self.layers[2]][:,:int(self.channel_8 / 4)]
        y_8 = self.conv_8_1(y_8)
        y_8 = self.conv_8_2(y_8)
        y_4 = outputs[self.layers[3]][:,:int(self.channel_4 / 4)]
        y_4 = self.conv_4_1(y_4)
        y_4 = self.conv_4_1(y_4)
        y_4 = self.conv_4_2(y_4)
        return torch.cat((y_32,y_16,y_8,y_4), 1)

class shuffle_2(nn.Module):
    def __init__(self, layers,channel_32,channel_16,channel_8,channel_4):
        super(shuffle_2, self).__init__()
        self.layers = layers  # layer indices
        self.channel_32=channel_32
        self.channel_16=channel_16
        self.channel_8=channel_8
        self.channel_4=channel_4
        # self.multiple = len(layers) > 1  # multiple layers flag
        self.conv_32 = nn.Conv2d(in_channels=int(channel_32/4), out_channels=int(channel_16/4), kernel_size=3, padding=3 // 2)
        self.upsample_32=nn.Upsample(scale_factor=2)
        # self.conv_16 = nn.Conv2d(in_channels=int(channel_16 / 4), out_channels=int(channel_16 / 4), kernel_size=3,stride=2,padding=3 // 2)
        self.conv_8 = nn.Conv2d(in_channels=int(channel_8/4),out_channels=int(channel_16/4),kernel_size=3,stride=2,padding=3//2)
        self.conv_4_1 = nn.Conv2d(in_channels=int(channel_4 / 4), out_channels=int(channel_4 / 4), kernel_size=3, stride=2, padding=3 // 2)
        self.conv_4_2 = nn.Conv2d(in_channels=int(channel_4 / 4), out_channels=int(channel_16 / 4), kernel_size=3,stride=2, padding=3 // 2)
    def forward(self, x,outputs):
        y_32=self.conv_32(outputs[self.layers[0]][:,int(self.channel_32/4):int(self.channel_32/2)])
        y_32=self.upsample_32(y_32)
        # print(y_32.shape)
        # print(y_32.shape)
        y_16 = outputs[self.layers[1]][:,int(self.channel_16/4):int(self.channel_16/2)]
        # y_16=self.conv_16(y_16)
        y_8 = outputs[self.layers[2]][:,int(self.channel_8 / 4):int(self.channel_8 / 2)]
        y_8 = self.conv_8(y_8)
        y_4 = outputs[self.layers[3]][:,int(self.channel_4 / 4):int(self.channel_4 / 2)]
        y_4 = self.conv_4_1(y_4)
        y_4 = self.conv_4_2(y_4)
        return torch.cat((y_32,y_16,y_8,y_4), 1)

class shuffle_3(nn.Module):
    def __init__(self, layers,channel_32,channel_16,channel_8,channel_4):
        super(shuffle_3, self).__init__()
        self.layers = layers  # layer indices
        self.channel_32=channel_32
        self.channel_16=channel_16
        self.channel_8=channel_8
        self.channel_4=channel_4
        # self.multiple = len(layers) > 1  # multiple layers flag
        self.conv_32 = nn.Conv2d(in_channels=int(channel_32/4), out_channels=int(channel_8/4), kernel_size=3, padding=3 // 2)
        self.upsample_32=nn.Upsample(scale_factor=4)
        self.conv_16 = nn.Conv2d(in_channels=int(channel_16/4), out_channels=int(channel_8/4), kernel_size=3, padding=3 // 2)
        self.upsample_16=nn.Upsample(scale_factor=2)
        # self.conv_16 = nn.Conv2d(in_channels=int(channel_16 / 4), out_channels=int(channel_16 / 4), kernel_size=3,stride=2,padding=3 // 2)
        # self.conv_8=nn.Conv2d(in_channels=int(channel_8/4),out_channels=int(channel_8/4),kernel_size=3,stride=2,padding=3//2)
        self.conv_4 = nn.Conv2d(in_channels=int(channel_4 / 4), out_channels=int(channel_8 / 4), kernel_size=3,
                                stride=2, padding=3 // 2)
    def forward(self, x,outputs):
        y_32=self.conv_32(outputs[self.layers[0]][:,int(self.channel_32/2):int(self.channel_32*0.75)])
        y_32=self.upsample_32(y_32)
        # print(y_32.shape)
        # print(y_32.shape)
        y_16 = self.conv_16(outputs[self.layers[1]][:,int(self.channel_16/2):int(self.channel_16*0.75)])
        y_16=self.upsample_16(y_16)
        y_8 = outputs[self.layers[2]][:,int(self.channel_8 / 2):int(self.channel_8*0.75)]
        # y_8 = self.conv_8(y_8)
        y_4 = outputs[self.layers[3]][:,int(self.channel_4 / 2):int(self.channel_4*0.75)]
        y_4= self.conv_4(y_4)
        return torch.cat((y_32,y_16,y_8,y_4), 1)

class shuffle_4(nn.Module):
    def __init__(self, layers,channel_32,channel_16,channel_8,channel_4):
        super(shuffle_4, self).__init__()
        self.layers = layers  # layer indices
        self.channel_32=channel_32
        self.channel_16=channel_16
        self.channel_8=channel_8
        self.channel_4=channel_4
        # self.multiple = len(layers) > 1  # multiple layers flag
        self.conv_32 = nn.Conv2d(in_channels=int(channel_32/4), out_channels=int(channel_4/4), kernel_size=9, padding=9 // 2)
        self.upsample_32=nn.Upsample(scale_factor=8)
        self.conv_16 = nn.Conv2d(in_channels=int(channel_16/4), out_channels=int(channel_4/4), kernel_size=3,padding=3 // 2)
        self.upsample_16=nn.Upsample(scale_factor=4)
        self.conv_8 = nn.Conv2d(in_channels=int(channel_8/4), out_channels=int(channel_4/4), kernel_size=3,padding=3 // 2)
        self.upsample_8=nn.Upsample(scale_factor=2)
        # self.conv_4 = nn.Conv2d(in_channels=int(channel_4 / 4), out_channels=int(channel_4 / 4), kernel_size=3,stride=2, padding=3 // 2)
    def forward(self, x,outputs):
        # y_32=self.conv_32(outputs[self.layers[0]][:,int(self.channel_32*0.75):self.channel_32])
        # y_32=self.upsample_32(y_32)
        # print(y_32.shape)
        # print(y_32.shape)
        y_16 = self.conv_16(outputs[self.layers[1]][:,int(self.channel_16*0.75):self.channel_16])
        y_16=self.upsample_16(y_16)
        y_8 =self.conv_8(outputs[self.layers[2]][:,int(self.channel_8*0.75):self.channel_8])
        y_8 = self.upsample_8(y_8)
        y_4 = outputs[self.layers[3]][:,int(self.channel_4*0.75):self.channel_4]
        # y_4= self.conv_4(y_4)
        return torch.cat((y_16,y_8,y_4), 1)
'''

class WeightedFeatureFusion(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers, weight=False):
        super(WeightedFeatureFusion, self).__init__()
        self.layers = layers  # layer indices
        self.weight = weight  # apply weights boolean
        self.n = len(layers) + 1  # number of layers
        if weight:
            self.w = nn.Parameter(torch.zeros(self.n), requires_grad=True)  # layer weights

    def forward(self, x, outputs):
        # Weights
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)  # sigmoid weights (0-1)
            x = x * w[0]

        # Fusion
        nx = x.shape[1]  # input channels
        for i in range(self.n - 1):
            a = outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[self.layers[i]]  # feature to add
            na = a.shape[1]  # feature channels

            # Adjust channels
            if nx == na:  # same shape
                x = x + a
            elif nx > na:  # slice input
                x[:, :na] = x[:, :na] + a  # or a = nn.ZeroPad2d((0, 0, 0, 0, 0, dc))(a); x = x + a
            else:  # slice feature
                x = x + a[:, :nx]

        return x


class MixConv2d(nn.Module):  # MixConv: Mixed Depthwise Convolutional Kernels https://arxiv.org/abs/1907.09595
    def __init__(self, in_ch, out_ch, k=(3, 5, 7), stride=1, dilation=1, bias=True, method='equal_params'):
        super(MixConv2d, self).__init__()

        groups = len(k)
        if method == 'equal_ch':  # equal channels per group
            i = torch.linspace(0, groups - 1E-6, out_ch).floor()  # out_ch indices
            ch = [(i == g).sum() for g in range(groups)]
        else:  # 'equal_params': equal parameter count per group
            b = [out_ch] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            ch = np.linalg.lstsq(a, b, rcond=None)[0].round().astype(int)  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([nn.Conv2d(in_channels=in_ch,
                                          out_channels=ch[g],
                                          kernel_size=k[g],
                                          stride=stride,
                                          padding=k[g] // 2,  # 'same' pad
                                          dilation=dilation,
                                          bias=bias) for g in range(groups)])

    def forward(self, x):
        return torch.cat([m(x) for m in self.m], 1)


# Activation functions below -------------------------------------------------------------------------------------------
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sx = torch.sigmoid(x)  # sigmoid(ctx)
        return grad_output * (sx * (1 + x * (1 - sx)))


class MishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sx = torch.sigmoid(x)
        fx = F.softplus(x).tanh()
        return grad_output * (fx + x * sx * (1 - fx * fx))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class MemoryEfficientMish(nn.Module):
    def forward(self, x):
        return MishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class HardSwish(nn.Module):  # https://arxiv.org/pdf/1905.02244.pdf
    def forward(self, x):
        return x * F.hardtanh(x + 3, 0., 6., True) / 6.


class Mish(nn.Module):  # https://github.com/digantamisra98/Mish
    def forward(self, x):
        return x * F.softplus(x).tanh()
