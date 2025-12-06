import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =============================================================================
# [Helper Modules]
# =============================================================================

class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM).
    Computes: (1/H*W * sum(x^p))^(1/p)
    - p=1 -> Average Pooling
    - p=inf -> Max Pooling
    """
    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W)
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3.0, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"

# =============================================================================
# [GhostNet Backbone Components]
# =============================================================================

def _make_divisible(v, divisor, min_value=None):
    """
    This function ensures that all layers have a channel number that is divisible by 8
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.PReLU(init_channels) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.PReLU(new_channels) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]

class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        self.conv = nn.Sequential(
            # pw
            GhostModule(inp, hidden_dim, kernel_size=1, relu=True),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            # Squeeze-and-Excitation would go here, but Strategy specified "GhostNet 1.3x" 
            # and later mentions CBAM or standard GhostNet. 
            # Standard GhostNet uses SE. We stick to standard SE for stability as per Hacker's implicit "GhostNet 1.3x" call.
            nn.Sequential() if not use_se else nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, hidden_dim // 4, 1, bias=True), # SE reduction 4
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 4, hidden_dim, 1, bias=True),
                nn.Hardsigmoid(inplace=True), # GhostNet uses Hardsigmoid
            ),
        )
        # pw-linear
        self.conv.add_module('ghost_proj', GhostModule(hidden_dim, oup, kernel_size=1, relu=False))

        self.shortcut = nn.Sequential()
        if stride == 2 or inp != oup:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride=stride, padding=1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, oup, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        # GhostNet implementation applies SE after DW, then PW-Linear
        # Re-implementing forward to inject SE correctly between DW and PW-Linear if using Sequential is tricky
        # So we reconstruct the forward explicitly for clarity
        return self.conv(x) + self.shortcut(x) 

    # To fix the SE injection above, let's redefine __init__ and forward slightly
    # Re-writing class for cleaner SE integration
class GhostBottleneckRefined(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super(GhostBottleneckRefined, self).__init__()
        self.use_se = use_se

        # Point-wise expansion
        self.ghost1 = GhostModule(inp, hidden_dim, kernel_size=1, relu=True)

        # Depth-wise
        self.dw_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False)
        self.dw_bn = nn.BatchNorm2d(hidden_dim)

        # SE Block
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, hidden_dim // 4, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 4, hidden_dim, 1, bias=True),
                nn.Hardsigmoid(inplace=True),
            )

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
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.ghost1(x)
        x = self.dw_conv(x)
        x = self.dw_bn(x)
        if self.use_se:
            x = x * self.se(x)
        x = self.ghost2(x)
        return x + residual

class GhostNetBackbone(nn.Module):
    def __init__(self, width_mult=1.3):
        super(GhostNetBackbone, self).__init__()
        # Standard GhostNet Config
        # k, t, c, SE, s
        cfgs = [
            # Stage 1
            [3,  16,  16, 0, 1],
            # Stage 2
            [3,  48,  24, 0, 2],
            [3,  72,  24, 0, 1],
            # Stage 3
            [5,  72,  40, 1, 2],
            [5, 120,  40, 1, 1],
            # Stage 4
            [3, 240,  80, 0, 2],
            [3, 200,  80, 0, 1],
            [3, 184,  80, 0, 1],
            [3, 184,  80, 0, 1],
            [3, 480, 112, 1, 1],
            [3, 672, 112, 1, 1],
            # Stage 5
            [5, 672, 160, 1, 2],
            [5, 960, 160, 0, 1],
            [5, 960, 160, 1, 1],
            [5, 960, 160, 0, 1],
            [5, 960, 160, 1, 1]
        ]

        # Building stem
        output_channel = _make_divisible(16 * width_mult, 4)
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, output_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.PReLU(output_channel)
        )
        input_channel = output_channel

        # Building blocks
        layers = []
        for k, exp_size, c, use_se, s in cfgs:
            output_channel = _make_divisible(c * width_mult, 4)
            hidden_channel = _make_divisible(exp_size * width_mult, 4)
            layers.append(GhostBottleneckRefined(input_channel, hidden_channel, output_channel, k, s, use_se))
            input_channel = output_channel
        self.blocks = nn.Sequential(*layers)

        # Building last expansion layer
        # Standard GhostNet expands to 960 (before width mult)
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
# [Head & Main Model]
# =============================================================================

class MetricHead(nn.Module):
    """
    Metric Learning Head (ArcFace/CosFace compatible).
    Computes Cosine Similarity: cos(theta) = (x . W) / (|x| . |W|)
    Forced FP32 execution for stability.
    """
    def __init__(self, in_features, num_classes):
        super(MetricHead, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # Force Float32 to prevent NaN in cosine calculation
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()
            W = self.weight.float()

            # Normalize inputs and weights
            x_norm = F.normalize(x, p=2, dim=1)
            W_norm = F.normalize(W, p=2, dim=1)

            # Compute cosine similarity (logits)
            # Output: (B, num_classes)
            logits = F.linear(x_norm, W_norm)

        return logits

class CustomModel(nn.Module):
    """
    Strategy Compliant Model:
    1. Backbone: GhostNet (Width=1.3x) with PReLU.
    2. Neck: GeM -> Flatten -> Dropout(0.2) -> Linear(1024) -> BatchNorm1d.
    3. Head: Metric Head (Cosine Similarity, FP32).
    """
    def __init__(self, num_classes=2882, img_size=256, embedding_size=1024):
        super(CustomModel, self).__init__()
        self.img_size = img_size

        # 1. Backbone (GhostNet 1.3x)
        self.backbone = GhostNetBackbone(width_mult=1.3)

        # 2. Neck
        # GeM Pooling
        self.gem = GeM(p=3.0)
        # Flatten is implicit
        self.dropout = nn.Dropout(p=0.2)

        # Embedding Layer
        # Calculate backbone output channels
        # GhostNet 1.3x: 960 * 1.3 ~= 1248
        backbone_out = self.backbone.out_channels

        self.fc = nn.Linear(backbone_out, embedding_size, bias=False)
        self.bn1d = nn.BatchNorm1d(embedding_size)

        # 3. Head
        self.head = MetricHead(embedding_size, num_classes)

        self._initialize_weights()

    def forward(self, x):
        # Features: (B, C, H, W)
        x = self.backbone(x)

        # Neck: GeM -> Flatten -> Dropout -> Linear -> BN
        x = self.gem(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        embedding = self.bn1d(x)

        # Head: Cosine Logits
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
        num_classes = 2882

    # img_size is passed for compatibility, though GhostNet handles arbitrary sizes 
    # (down to a limit) due to pooling.
    return CustomModel(num_classes=num_classes, img_size=img_size, embedding_size=1024)