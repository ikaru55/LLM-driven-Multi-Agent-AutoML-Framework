import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =============================================================================
# [Helper Functions]
# =============================================================================

def _make_divisible(v, divisor, min_value=None):
    """
    Ensures that all layers have a channel number divisible by the divisor (usually 8).
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# =============================================================================
# [Building Blocks] GhostModule, CoordAtt, GeM
# =============================================================================

class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM).
    Computes: (1/H*W * sum(x^p))^(1/p)
    - p=1 -> Average Pooling
    - p=inf -> Max Pooling
    """
    def __init__(self, p=3.0, eps=1e-06):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3.0, eps=1e-06):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})'


class CoordAtt(nn.Module):
    """
    Coordinate Attention for Efficient Mobile Network Design.
    Factorizes channel attention into two 1D feature encodings to capture spatial structure.
    """
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # 1. Feature Aggregation (H and W directions)
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2) # [N, C, W, 1]

        # 2. Concatenation & Transformation
        y = torch.cat([x_h, x_w], dim=2) # [N, C, H+W, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 

        # 3. Split & Attentional Weights
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # 4. Re-weighting
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
            nn.PReLU(init_channels) if relu else nn.Sequential()
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.PReLU(new_channels) if relu else nn.Sequential()
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

        # Point-wise expansion
        self.ghost1 = GhostModule(inp, hidden_dim, kernel_size=1, relu=True)

        # Depth-wise convolution
        if stride > 1:
            self.dw_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False)
            self.dw_bn = nn.BatchNorm2d(hidden_dim)
        else:
            self.dw_conv = nn.Identity()
            self.dw_bn = nn.Identity()

        # Coordinate Attention (replaces SE)
        if use_att:
            self.att = CoordAtt(hidden_dim, hidden_dim)
        else:
            self.att = nn.Identity()

        # Point-wise linear projection
        self.ghost2 = GhostModule(hidden_dim, oup, kernel_size=1, relu=False)

        # Shortcut
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

# =============================================================================
# [Backbone] GhostNet 1.3x
# =============================================================================

class GhostNetBackbone(nn.Module):
    def __init__(self, width_mult=1.3):
        super(GhostNetBackbone, self).__init__()

        # Configuration: [kernel, exp_size, out_channels, use_att, stride]
        # use_att = 1 indicates usage of Coordinate Attention
        cfgs = [
            # Stage 1
            [3, 16, 16, 0, 1],
            # Stage 2
            [3, 48, 24, 0, 2],
            [3, 72, 24, 0, 1],
            # Stage 3
            [5, 72, 40, 1, 2],
            [5, 120, 40, 1, 1],
            # Stage 4
            [3, 240, 80, 0, 2],
            [3, 200, 80, 0, 1],
            [3, 184, 80, 0, 1],
            [3, 184, 80, 0, 1],
            [3, 480, 112, 1, 1],
            [3, 672, 112, 1, 1],
            # Stage 5
            [5, 672, 160, 1, 2],
            [5, 960, 160, 0, 1],
            [5, 960, 160, 1, 1],
            [5, 960, 160, 0, 1],
            [5, 960, 160, 1, 1]
        ]

        # Building first layer
        output_channel = _make_divisible(16 * width_mult, 4)
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, output_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.PReLU(output_channel)
        )

        input_channel = output_channel
        layers = []
        for k, exp_size, c, use_att, s in cfgs:
            output_channel = _make_divisible(c * width_mult, 4)
            hidden_channel = _make_divisible(exp_size * width_mult, 4)
            layers.append(GhostBottleneck(input_channel, hidden_channel, output_channel, k, s, use_att))
            input_channel = output_channel

        self.blocks = nn.Sequential(*layers)

        # Building last several layers
        exp_channel = _make_divisible(960 * width_mult, 4)
        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channel, exp_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(exp_channel),
            nn.PReLU(exp_channel)
        )
        self.out_channels = exp_channel

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.blocks(x)
        x = self.conv_last(x)
        return x

# =============================================================================
# [Head] Metric Learning Head
# =============================================================================

class MetricHead(nn.Module):
    """
    Metric Learning Head (Linear Layer for Cosine Similarity).
    Calculates logits = Scale * Cosine(x, W).
    The Scale (s) is typically handled in the Loss function, but the Logits
    here are the raw dot products of normalized vectors.

    IMPORTANT: The loss function in trainer_logic.py (MagFace/ElasticFace) expects
    logits that can be processed. Usually, the 's' factor is applied inside the 
    loss or at the very end. The standard interface here outputs 'cosine', 
    and the loss applies 's'. However, to be safe and compatible with standard 
    CrossEntropy if needed, we output the raw projection. 

    The Trainer Logic (MagFace/ElasticFace) applies 's' internally.
    So this module outputs raw Cosine Similarity if we normalize here, 
    or raw Logits if we don't.

    Standard Practice for ArcFace/MagFace implementations:
    - Forward returns: cosine (shape: B, NumClasses)
    - Loss calculates: s * cosine -> Apply margin -> Softmax
    """
    def __init__(self, in_features, num_classes):
        super(MetricHead, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # Normalize features and weights to project onto the hypersphere
        # Force float32 for precision in metric learning
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()
            W = F.normalize(self.weight.float(), p=2, dim=1)
            x = F.normalize(x, p=2, dim=1)
            cosine = F.linear(x, W)
        return cosine

# =============================================================================
# [Main Model] CustomModel
# =============================================================================

class CustomModel(nn.Module):
    """
    Refined Architecture:
    1. Backbone: GhostNet 1.3x with Coordinate Attention.
    2. Neck: GeM -> Flatten -> Linear(512) -> BN1d.
    3. Head: Metric Head (Cosine).

    Constraints:
    - Embedding Size: 512 (Compact)
    - Params: < 11.5M
    """
    def __init__(self, num_classes=2882, img_size=112, embedding_size=512):
        super(CustomModel, self).__init__()
        self.img_size = img_size

        # 1. Backbone
        self.backbone = GhostNetBackbone(width_mult=1.3)

        # 2. Neck
        # GeM Pooling
        self.gem = GeM(p=3.0)

        # Projection Layer (Backbone Out -> 512)
        backbone_out = self.backbone.out_channels
        self.fc = nn.Linear(backbone_out, embedding_size, bias=False)
        self.bn1d = nn.BatchNorm1d(embedding_size)

        # 3. Head
        self.head = MetricHead(embedding_size, num_classes)

        self._initialize_weights()

    def forward(self, x):
        # [CRITICAL] Input: x only. Output: Logits (B, num_classes).

        # 1. Backbone features
        x = self.backbone(x)

        # 2. Neck (Pooling -> Linear -> BN)
        x = self.gem(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        embedding = self.bn1d(x)

        # 3. Head (Cosine Similarity)
        # Note: 's' (scale) is applied in the Loss function (Trainer Logic)
        logits = self.head(embedding)

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

# =============================================================================
# [Interface] Factory Function
# =============================================================================

def get_model(num_classes=None, img_size=112, **kwargs):
    """
    Factory function called by the framework.
    """
    if num_classes is None:
        num_classes = 2882 # Default fallback

    # Embedding size fixed to 512 as per Hacker's strategy
    return CustomModel(num_classes=num_classes, img_size=img_size, embedding_size=512)