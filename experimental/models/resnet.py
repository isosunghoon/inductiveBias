import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        layers,
        num_classes=100,
        stem_channels=64,
        base_channels=64,
        zero_init_residual=False,
    ):
        super().__init__()
        self.in_channels = stem_channels

        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )

        widths = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        strides = [1, 2, 2, 2]

        self.layer1 = self._make_layer(widths[0], layers[0], stride=strides[0])
        self.layer2 = self._make_layer(widths[1], layers[1], stride=strides[1])
        self.layer3 = self._make_layer(widths[2], layers[2], stride=strides[2])
        self.layer4 = self._make_layer(widths[3], layers[3], stride=strides[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(widths[3], num_classes)

        self._init_weights(zero_init_residual)

    def _make_layer(self, out_channels, blocks, stride):
        layers = [BasicBlock(self.in_channels, out_channels, stride=stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self, zero_init_residual):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, BasicBlock):
                    nn.init.constant_(module.bn2.weight, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


def build_resnet(args):
    layer_map = {
        "resnet18": [2, 2, 2, 2],
    }

    if args.model not in layer_map:
        raise ValueError(f"Unsupported ResNet model: {args.model}")

    if args.resnet_block != "basic":
        raise ValueError(f"Unsupported ResNet block: {args.resnet_block}")

    return ResNet(
        layers=layer_map[args.model],
        num_classes=args.num_classes,
        stem_channels=args.resnet_stem_channels,
        base_channels=args.resnet_base_channels,
        zero_init_residual=args.resnet_zero_init_residual,
    )
