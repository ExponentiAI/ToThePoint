import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from models.pointnet_utils import STN3d, STNkd, feature_transform_reguliarzer
import random

class Recycle_Dual_Point(nn.Module):
    def __init__(self):
        super(Recycle_Dual_Point, self).__init__()
    def forward(self,x):
        x_sort, indices = torch.sort(x,dim=2,descending =True)
        recyle_point = x_sort[:, :, random.randint(1, x_sort.shape[-1]-1)]
        return recyle_point#.detach()

class PointNet_Part_Seg(nn.Module):
    def __init__(self, part_num=50, normal_channel=True,args = None):
        super(PointNet_Part_Seg, self).__init__()
        self.pretrain = args.pretrain
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.part_num = part_num
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 2048, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)
        self.fstn = STNkd(k=128)
        self.convs1 = torch.nn.Conv1d(4944, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, part_num, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

        if self.pretrain:
            self.prototype_head = nn.Sequential(
                                nn.Linear(2048 , 512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(inplace=True),
                                nn.Linear(512, 256)
                                )

            self.recycle_dual_point  = Recycle_Dual_Point()

    def forward(self, point_cloud, label=None):
        B, D, N = point_cloud.size()
        trans = self.stn(point_cloud)
        point_cloud = point_cloud.transpose(2, 1)
        if D > 3:
            point_cloud, feature = point_cloud.split(3, dim=2)
        point_cloud = torch.bmm(point_cloud, trans)
        if D > 3:
            point_cloud = torch.cat([point_cloud, feature], dim=2)

        point_cloud = point_cloud.transpose(2, 1)

        out1 = F.relu(self.bn1(self.conv1(point_cloud)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))

        trans_feat = self.fstn(out3)
        x = out3.transpose(2, 1)
        net_transformed = torch.bmm(x, trans_feat)
        net_transformed = net_transformed.transpose(2, 1)

        out4 = F.relu(self.bn4(self.conv4(net_transformed)))
        out5 = self.bn5(self.conv5(out4))

        out_max = torch.max(out5, 2, keepdim=True)[0]
        out_max = out_max.view(-1, 2048)

        if self.pretrain:
            recylce_point = self.recycle_dual_point(out5)
            # max_point = torch.max(out5, 2, keepdim=False)[0]
            recylce_point = recylce_point.view(-1, 2048)
            max_point = out_max.view(-1, 2048)
            x1_prototype = self.prototype_head(out_max)
            x2_prototype = self.prototype_head(recylce_point)
            return max_point,x1_prototype,x2_prototype

        out_max = torch.cat([out_max,label.squeeze(1)],1)
        expand = out_max.view(-1, 2048+16, 1).repeat(1, 1, N)
        concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)
        net = F.relu(self.bns1(self.convs1(concat)))
        net = F.relu(self.bns2(self.convs2(net)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = self.convs4(net)
        net = net.transpose(2, 1).contiguous()
        net = F.log_softmax(net.view(-1, self.part_num), dim=-1)
        net = net.view(B, N, self.part_num) # [B, N, 50]

        return net, trans_feat


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss