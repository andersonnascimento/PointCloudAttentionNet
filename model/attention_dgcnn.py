import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utility import get_graph_feature
from utils.utility import knn


class AttentionDGCNN(nn.Module):
    def __init__(self, args):
        super(AttentionDGCNN, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.bn1_att = nn.BatchNorm1d(64)
        self.bn2_att = nn.BatchNorm1d(64)
        self.bn3_att = nn.BatchNorm1d(128)
        self.bn4_att = nn.BatchNorm1d(256)

        self.attn1 = nn.MultiheadAttention(64, args.att_heads)
        self.attn2 = nn.MultiheadAttention(64, args.att_heads)
        self.attn3 = nn.MultiheadAttention(128, args.att_heads)
        self.attn4 = nn.MultiheadAttention(256, args.att_heads)

        self.conv1_l1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv1_l2 = self.bn1

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(256*2, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, args.number_classes)

    def forward(self, x):
        batch_size = x.size(0)

        x = get_graph_feature(x, k=self.k)      #[32, 6, 1024, 40]
        x = self.conv1(x)                       #[32, 64, 1024, 40]
        x1 = x.max(dim=-1, keepdim=False)[0]    #[32, 64, 1024]

        residual = x1
        x1_T = x1.transpose(1, 2)               #[32, 1024, 64]
        x1_att, _ = self.attn1(x1_T,x1_T,x1_T)  #[32, 1024, 64]
        x1 = x1_att.transpose(1,2)              #[32, 64, 1024]
        del x1_T, x1_att, _
        x1 += residual

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        residual = x2
        x2_T = x2.transpose(1, 2)
        x2_att, _ = self.attn2(x2_T,x2_T,x2_T)
        x2 = x2_att.transpose(1,2)
        del x2_T, x2_att, _
        x2 += residual

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        residual = x3
        x3_T = x3.transpose(1, 2)
        x3_att, _ = self.attn3(x3_T,x3_T,x3_T)
        x3 = x3_att.transpose(1,2)
        del x3_T, x3_att, _
        x3 += residual

        x = get_graph_feature(x3, k=self.k) #uncomment to disable attention
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        residual = x4
        x4_T = x4.transpose(1, 2)
        x4_att, _ = self.attn4(x4_T,x4_T,x4_T)
        x4 = x4_att.transpose(1,2)
        del x4_T, x4_att, _
        x4 += residual

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x
