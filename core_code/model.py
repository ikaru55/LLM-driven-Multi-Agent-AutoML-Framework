import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class h_swish(nn.Module):

    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    """
    Coordinate Attention (CVPR 2021)
    Factorizes attention into two 1D feature encoding processes.
    """

    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_h * a_w
        return out

class Bottleneck(nn.Module):

    def __init__(self, inp, oup, stride, expansion, use_ca=False):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup
        exp_size = inp * expansion
        layers = []
        if expansion != 1:
            layers.append(nn.Conv2d(inp, exp_size, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(exp_size))
            layers.append(nn.PReLU(exp_size))
        layers.append(nn.Conv2d(exp_size, exp_size, 3, stride, 1, groups=exp_size, bias=False))
        layers.append(nn.BatchNorm2d(exp_size))
        layers.append(nn.PReLU(exp_size))
        if use_ca:
            layers.append(CoordAtt(exp_size, exp_size))
        layers.append(nn.Conv2d(exp_size, oup, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(oup))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class ConvBlock(nn.Module):

    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp if dw else 1, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        self.linear = linear
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if not self.linear:
            x = self.prelu(x)
        return x

class MobileFaceNetV2(nn.Module):

    def __init__(self, embedding_size=512, width_mult=2.5):
        super(MobileFaceNetV2, self).__init__()
        self.cfgs = [[2, 64, 5, 2, False], [4, 128, 1, 2, True], [2, 128, 6, 1, True], [4, 128, 1, 2, True], [2, 128, 2, 1, True]]
        input_channel = int(64 * width_mult)
        self.conv1 = ConvBlock(3, input_channel, 3, 2, 1)
        layers = []
        for t, c, n, s, ca in self.cfgs:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(Bottleneck(input_channel, output_channel, stride, t, use_ca=ca))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        self.conv2 = ConvBlock(input_channel, 512, 1, 1, 0)
        self.linear7x7 = ConvBlock(512, 512, 7, 1, 0, dw=True, linear=True)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(512, embedding_size, bias=False)
        self.bn1 = nn.BatchNorm1d(embedding_size)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.conv2(x)
        x = self.linear7x7(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.bn1(x)
        return x

class MetricHead(nn.Module):
    """
    Implements the cosine similarity layer.
    Returns: Scaled Logits (s * cos_theta)
    """

    def __init__(self, embedding_size, num_classes, s=64.0):
        super(MetricHead, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        self.s = s
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding):
        embedding_norm = F.normalize(embedding, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embedding_norm, weight_norm)
        output = cosine * self.s
        return output

class CustomModel(nn.Module):

    def __init__(self, num_classes=2882, img_size=112):
        super(CustomModel, self).__init__()
        self.img_size = img_size
        self.backbone = MobileFaceNetV2(embedding_size=512, width_mult=2.5)
        self.head = MetricHead(embedding_size=512, num_classes=num_classes, s=64.0)

    def forward(self, x):
        embedding = self.backbone(x)
        logits = self.head(embedding)
        return logits

def get_model(num_classes=None, img_size=112, **kwargs):
    """
    Factory function called by the framework.
    """
    if num_classes is None:
        num_classes = 2882
    return CustomModel(num_classes=num_classes, img_size=img_size)