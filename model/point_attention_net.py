import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.utility import get_graph_feature
from utils.utility import knn

class PointAttentionNet(nn.Module):
    def __init__(self, args):
        super(PointAttentionNet, self).__init__()
        self.args = args

        self.attn1 = nn.MultiheadAttention(64, args.att_heads)
        self.attn2 = nn.MultiheadAttention(64, args.att_heads)
        self.attn3 = nn.MultiheadAttention(128, args.att_heads)

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, args.number_classes)

    def perform_att(self, att, x):
        residual = x
        # print(x.shape)

        x_T = x.transpose(1, 2)
        x_att, _ = att(x_T,x_T,x_T)
        x = x_att.transpose(1,2)
        # x, _ = att(x,x,x)

        return x + residual

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.perform_att(self.attn1, self.conv2(x))))
        x = F.relu(self.bn3(self.perform_att(self.attn2, self.conv3(x))))
        x = F.relu(self.bn4(self.perform_att(self.attn3, self.conv4(x))))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x
