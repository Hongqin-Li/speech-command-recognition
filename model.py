import torch
import torch.nn as nn


class VGG1d(nn.Module):
    def __init__(self, features, num_classes, hidden_size=128):
        super(VGG1d, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool1d(3)
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3, hidden_size),
            nn.ReLU(True),
            nn.Dropout(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


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
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 128, 'M'],
}


def vgg1d_bn(in_channels, **kwargs):
    return VGG1d(make_layers1d(cfgs['A'], in_channels, batch_norm=True),
                 **kwargs)
