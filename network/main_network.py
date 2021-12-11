from network.darknet import *
import torch


class ModelMain(nn.Module):
    def __init__(self, class_num):
        super(ModelMain, self).__init__()
        self.class_num = class_num
        self.bbox_num = 3
        self.loc_conf_num = 5

        self.out_channels = self.bbox_num * (self.loc_conf_num + self.class_num)  # coco dataset has 12 categories

        self.darknet53 = DarkNet([1, 2, 8, 8, 4])
        self.conv_11_0 = self.con_1_1(in_=1024, out_=self.out_channels)  # tx,ty,w,h,p,classes 3 bounding box
        self.conv_11_1 = self.con_1_1(in_=512, out_=self.out_channels)  # tx,ty,w,h,p,classes 3 bounding box
        self.conv_11_2 = self.con_1_1(in_=256, out_=self.out_channels)  # tx,ty,w,h,p,classes 3 bounding box

        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')

        self.con_1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.con_2 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)

        self.embedding13 = self.cob_block(in_channel=1024, channel_list=[512, 1024])
        self.embedding26 = self.cob_block(in_channel=768, channel_list=[256, 512])
        self.embedding52 = self.cob_block(in_channel=384, channel_list=[128, 256])

    def forward(self, x):
        out52, out26, out13 = self.darknet53(x)

        x_13 = self.embedding13(out13)
        branch_1 = self.upsample1(x_13)
        branch_1 = self.con_1(branch_1)
        com_1 = torch.cat([out26, branch_1], dim=1)
        x_13 = self.conv_11_0(x_13)

        x_26 = self.embedding26(com_1)
        branch_2 = self.upsample2(x_26)
        branch_2 = self.con_2(branch_2)
        com_2 = torch.cat([out52, branch_2], dim=1)
        x_26 = self.conv_11_1(x_26)

        x_52 = self.embedding52(com_2)
        x_52 = self.conv_11_2(x_52)

        return x_13, x_26, x_52

    def bblock(self, in_, out_, ks):
        pad = (ks - 1) // 2 if ks else 0
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(in_, out_, kernel_size=ks, stride=1, padding=pad, bias=False)),
            ("bn", nn.BatchNorm2d(out_)),
            ("relu", nn.LeakyReLU(0.1))
        ]))

    def con_1_1(self, in_, out_):
        m = nn.Conv2d(in_channels=in_, out_channels=out_, kernel_size=1, stride=1, padding=0, bias=False)
        return m

    def cob_block(self, in_channel, channel_list):
        model = nn.ModuleList([
            self.bblock(in_=in_channel, out_=channel_list[0], ks=1),
            self.bblock(in_=channel_list[0], out_=channel_list[1], ks=3),
            self.bblock(in_=channel_list[1], out_=channel_list[0], ks=1),
            self.bblock(in_=channel_list[0], out_=channel_list[1], ks=3),
            self.bblock(in_=channel_list[1], out_=channel_list[0], ks=1),
            self.bblock(in_=channel_list[0], out_=channel_list[1], ks=3),
        ])
        return nn.Sequential(*model)

# x = torch.randn(1,3,416,416)
# m = ModelMain()
# o_13,o_26,o_52 = m(x)
# print(o_13.size(),o_26.size(),o_52.size())
