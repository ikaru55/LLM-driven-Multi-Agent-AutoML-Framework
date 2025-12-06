import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =============================================================================
# [Model Architect's Notes]
# Implements the "Hacker's Refined Strategy" (Cycle 18+):
# 1. Backbone: GhostNet 1.0x (Upgraded from 0.75x) for better capacity (~6M params).
# 2. Activation: HardSwish replaced PReLU for numerical stability and preventing gradient explosion.
# 3. Neck: GDC (Global Depthwise Convolution) replaced GeM.
#    - Preserves spatial centrality.
#    - Fixed kernel size=4 for 112x112 input (Output 4x4).
# 4. Head: MetricHead outputs scaled logits (s=64.0) to match Trainer's MagFaceLoss expectations.
# 5. Total Params: ~8-9M (< 11.5M Constraint).
# =============================================================================

def _make_divisible(v, divisor, min_value=None):
    """
    Ensures that all layers have a channel number divisible by the divisor (usually 8).
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class HardSwish(nn.Module):
    """
    HardSwish (x * ReLU6(x+3)/6) is more numerically stable than PReLU
    and parameter-free, reducing the risk of divergence.
    """
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return x * self.relu6(x + 3.0) / 6.0

class CoordAtt(nn.Module):
    """
    Coordinate Attention: Captures long-range dependencies with precise positional information.
    Modified to use HardSwish.
    """
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = HardSwish()

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

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            HardSwish() if relu else nn.Sequential()
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            HardSwish() if relu else nn.Sequential()
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]

class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_att):
        super(GhostBottleneck, self).__init__()
        self.use_att = use_att

        self.ghost1 = GhostModule(inp, hidden_dim, kernel_size=1, relu=True)

        if stride > 1:
            self.dw_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False)
            self.dw_bn = nn.BatchNorm2d(hidden_dim)
        else:
            self.dw_conv = nn.Identity()
            self.dw_bn = nn.Identity()

        if use_att:
            self.att = CoordAtt(hidden_dim, hidden_dim)
        else:
            self.att = nn.Identity()

        self.ghost2 = GhostModule(hidden_dim, oup, kernel_size=1, relu=False)

        if stride == 1 and inp == oup:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride=stride, padding=1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, oup, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.ghost1(x)
        if not isinstance(self.dw_conv, nn.Identity):
            x = self.dw_conv(x)
            x = self.dw_bn(x)
        if self.use_att:
            x = self.att(x)
        x = self.ghost2(x)
        return x + residual

class GhostNetBackbone(nn.Module):
    def __init__(self, width_mult=1.0):
        super(GhostNetBackbone, self).__init__()
        # GhostNet config: [k, t, c, SE, s]
        cfgs = [
            [3, 16, 16, 0, 1], 
            [3, 48, 24, 0, 2], 
            [3, 72, 24, 0, 1], 
            [5, 72, 40, 1, 2], 
            [5, 120, 40, 1, 1], 
            [3, 240, 80, 0, 2], 
            [3, 200, 80, 0, 1], 
            [3, 184, 80, 0, 1], 
            [3, 184, 80, 0, 1], 
            [3, 480, 112, 1, 1], 
            [3, 672, 112, 1, 1], 
            [5, 672, 160, 1, 2], 
            [5, 960, 160, 0, 1], 
            [5, 960, 160, 1, 1], 
            [5, 960, 160, 0, 1], 
            [5, 960, 160, 1, 1]
        ]

        output_channel = _make_divisible(16 * width_mult, 4)
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, output_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channel),
            HardSwish()
        )

        input_channel = output_channel
        layers = []
        for k, exp_size, c, use_att, s in cfgs:
            output_channel = _make_divisible(c * width_mult, 4)
            hidden_channel = _make_divisible(exp_size * width_mult, 4)
            layers.append(GhostBottleneck(input_channel, hidden_channel, output_channel, k, s, use_att))
            input_channel = output_channel

        self.blocks = nn.Sequential(*layers)

        # Final stage
        exp_channel = _make_divisible(960 * width_mult, 4)
        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channel, exp_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(exp_channel),
            HardSwish()
        )
        self.out_channels = exp_channel

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.blocks(x)
        x = self.conv_last(x)
        return x

class GDC(nn.Module):
    """
    Global Depthwise Convolution (GDC) Neck.
    Replaces GeM/GAP. Treats the final spatial map (4x4) as a distinct structure.
    """
    def __init__(self, in_channels, embedding_size, kernel_size=4):
        super(GDC, self).__init__()
        # Depthwise Conv: kernel_size matches feature map size (4x4 for 112 input)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   groups=in_channels, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.act_dw = HardSwish()

        # Pointwise Conv: Reduces channels to embedding size
        self.pointwise = nn.Conv2d(in_channels, embedding_size, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(embedding_size)
        # Note: No activation after final projection in neck usually, 
        # but standard GDC block has BN-PReLU. We use BN here.

    def forward(self, x):
        # Fallback if input size is not 4x4 (e.g. variable resolution testing)
        if x.size(2) != self.depthwise.kernel_size[0] or x.size(3) != self.depthwise.kernel_size[1]:
            x = F.adaptive_avg_pool2d(x, self.depthwise.kernel_size)

        x = self.depthwise(x)
        x = self.bn_dw(x)
        x = self.act_dw(x)

        x = self.pointwise(x)
        x = self.bn_pw(x)
        return x.flatten(1)

class MetricHead(nn.Module):
    """
    Metric Learning Head (ArcFace/MagFace style).
    Output: Scaled Logits = s * Cosine(x, W)
    """
    def __init__(self, in_features, num_classes, s=64.0):
        super(MetricHead, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # x: (B, in_features)
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()
            W = F.normalize(self.weight.float(), p=2, dim=1)
            x = F.normalize(x, p=2, dim=1)
            cosine = F.linear(x, W)
            logits = cosine * self.s
        return logits

class CustomModel(nn.Module):
    """
    Full Architecture: GhostNet 1.0x -> GDC -> MetricHead
    """
    def __init__(self, num_classes=2882, img_size=112):
        super(CustomModel, self).__init__()
        self.img_size = img_size

        # 1. Backbone: GhostNet 1.0x (width_mult=1.0)
        # 112x112 input -> 4x4 feature map
        self.backbone = GhostNetBackbone(width_mult=1.0)

        # 2. Neck: GDC
        # Output of 1.0x backbone is typically 960 channels
        self.embedding_size = 512
        self.neck = GDC(in_channels=self.backbone.out_channels, 
                        embedding_size=self.embedding_size, 
                        kernel_size=4) # Fixed for 112 input

        # 3. Head: Metric Learning
        self.head = MetricHead(self.embedding_size, num_classes, s=64.0)

        self._initialize_weights()

    def forward(self, x):
        # x: (B, 3, 112, 112)
        features = self.backbone(x)     # -> (B, 960, 4, 4)
        embedding = self.neck(features) # -> (B, 512)
        logits = self.head(embedding)   # -> (B, num_classes)
        return logits

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

def get_model(num_classes=None, img_size=112, **kwargs):
    """
    Factory function called by the framework.
    """
    if num_classes is None:
        num_classes = 2882
    return CustomModel(num_classes=num_classes, img_size=img_size)