# -*- coding: utf-8 -*-


import torch.nn as nn
import torch
import torch.nn.functional as F

class Change_channel_1x1(nn.Module):
    def __init__(self,channel_list,inter_channel = 512):
        super(Change_channel_1x1, self).__init__()

        self.conv1 = nn.ModuleList()
        for i in range(len(channel_list)):
            conv1 = nn.Sequential(
                nn.Conv2d(channel_list[i] , inter_channel, kernel_size=1, stride=1),
                nn.BatchNorm2d(inter_channel),
                nn.ReLU(),)
            self.conv1.append(conv1)

    def forward(self, f_list):
        f_conv_list = []
        for i in  range(len(f_list)):
            x = self.conv1[i](f_list[i])
            f_conv_list.append(x)

        return f_conv_list


class Change_channel_1x1_up(nn.Module):
    def __init__(self, channel_list, inter_channel=512):
        super(Change_channel_1x1_up, self).__init__()

        self.conv1_1 = nn.ModuleList()
        for i in range(len(channel_list)):
            conv1_1 = nn.Sequential(
                nn.Conv2d(inter_channel, channel_list[i], kernel_size=1, stride=1),
                nn.BatchNorm2d(channel_list[i]),
                nn.ReLU(), )
            self.conv1_1.append(conv1_1)

    def forward(self, f_list):
        f_conv_list_up = []
        for i in range(len(f_list)):
            x = self.conv1_1[i](f_list[i])
            f_conv_list_up.append(x)

        return f_conv_list_up




class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CCFA(nn.Module):

    def __init__(self, F_f=[64,128,320,512]):
        super().__init__()


        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(512, 512))

        self.relu = nn.ReLU(inplace=True)
        self.change_1 = Change_channel_1x1(F_f)
        self.change_2 = Change_channel_1x1_up(F_f)

        self.conV = nn.Linear(512*4,512)

        self.ffn_norm1 = nn.BatchNorm2d(F_f[0])
        self.ffn_norm2 = nn.BatchNorm2d(F_f[1])
        self.ffn_norm3 = nn.BatchNorm2d(F_f[2])
        self.ffn_norm4 = nn.BatchNorm2d(F_f[3])

    def forward(self, f):

        stage_num  =  len(f)
        org_x1 = f[0]
        org_x2 = f[1]
        org_x3 = f[2]
        org_x4 = f[3]

        channel_att_x_list = []
        x_after_channel_list = []
        f = self.change_1(f)

        for i in range(stage_num):

            avg_pool_x = F.avg_pool2d( f[i], (f[i].size(2), f[i].size(3)), stride=(f[i].size(2), f[i].size(3)))
            channel_att_x = self.mlp_x(avg_pool_x)
            channel_att_x_list.append(channel_att_x)

        channel_att_cat = torch.cat(channel_att_x_list,dim=1)
        channel_att_cat = self.conV(channel_att_cat)


        for i in range(stage_num):
            scale = torch.sigmoid(channel_att_cat).unsqueeze(2).unsqueeze(3).expand_as(f[i])
            x_after_channel = f[i] * scale
            out = self.relu(x_after_channel)
            x_after_channel_list.append(out)

        f_after_list = self.change_2(x_after_channel_list)
        f_after_list[0] = f_after_list[0] + org_x1
        f_after_list[1] = f_after_list[1] + org_x2
        f_after_list[2] = f_after_list[2] + org_x3
        f_after_list[3] = f_after_list[3] + org_x4

        return f_after_list




if __name__ == '__main__':

    x1 = torch.rand(1, 64, 56, 56)
    x2 = torch.rand(1, 128, 28, 28)
    x3 = torch.rand(1, 320, 14, 14)
    x4 = torch.rand(1, 512, 7, 7)

    list = [x1,x2,x3,x4]


    net = CCFA()

    for i in range(0,4):
        print(net(list)[i].size())
