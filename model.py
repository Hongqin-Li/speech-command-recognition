import torch
import torch.nn as nn


class VGG1d(nn.Module):
    def __init__(self, features, num_classes, hidden_size=128):
        super(VGG1d, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool1d(3)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3, hidden_size),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VGG(nn.Module):

    def __init__(self, features, num_classes,
                 hidden_size=1024, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, hidden_size),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_size, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_layers1d(cfg, in_channels, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
        else:
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv1d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],

    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512,
          'M', 512, 512, 'M'],

    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,
          'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],

    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256,
          'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg1d_bn(in_channels, **kwargs):
    return VGG1d(make_layers1d(cfgs['A'], in_channels, batch_norm=True),
                 **kwargs)


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download
                         to stderr
    """
    return VGG(make_layers(cfgs['A'], batch_norm=True), **kwargs)
