import torch.nn as nn
from collections import OrderedDict


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BasicBlock, self).__init__()
        # image size will not change
        self.conv1 = nn.Conv2d(in_channel, out_channel[0], kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel[0])
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(out_channel[0], out_channel[1], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        x_skip = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += x_skip
        return out


class DarkNet(nn.Module):
    def __init__(self, layers_list):
        super(DarkNet, self).__init__()
        self.in_channel = 32
        self.layer0 = self.first_layer(self.in_channel)

        self.layer1 = self.make_layers([32, 64], layers_list[0])
        self.layer2 = self.make_layers([64, 128], layers_list[1])
        self.layer3 = self.make_layers([128, 256], layers_list[2])
        self.layer4 = self.make_layers([256, 512], layers_list[3])
        self.layer5 = self.make_layers([512, 1024], layers_list[4])

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)  # (batch_size,256,52,52)
        out4 = self.layer4(out3)  # (batch_size,512,26,26)
        out5 = self.layer5(out4)  # (batch_size,1024,13,13)
        return out3, out4, out5

    def make_layers(self, out_channel, blocks):
        '''

        :param out_channel:
        :param blocks:
        :return:
        '''
        layers = []
        layers.append(("ds_conv", nn.Conv2d(self.in_channel, out_channel[1],kernel_size=3,stride=2,padding=1,bias=False)))
        layers.append(("ds_batchnorm", nn.BatchNorm2d(out_channel[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))

        self.in_channel = out_channel[1]
        for i in range(blocks):
            layers.append(("Basic_Block_{}".format(i), BasicBlock(self.in_channel, out_channel)))
        return nn.Sequential(OrderedDict(layers))

    def first_layer(self, out_channel):
        '''
        in_w,in_h don't change/ channel num changed 3->32
        :param out_channel:
        :return:
        '''
        layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.1)
        )
        return layer


