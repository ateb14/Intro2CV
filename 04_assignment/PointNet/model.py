from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

# ----------TODO------------
# Implement the PointNet 
# ----------TODO------------

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, d=1024):
        super(PointNetfeat, self).__init__()

        self.d = d
        self.global_feat = global_feat

        self.first_layer = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        self.mlp = nn.Sequential(
            # input shape of linear layer: (batch_size, n, 64)
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, d, 1),
            nn.BatchNorm1d(d),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        
        n_pts = x.shape[-1]
        x = self.first_layer(x)

        if self.global_feat:
            x = self.mlp(x)
            vis_feat = x.reshape(-1, n_pts, self.d)
            out = torch.max(x, -1, keepdim=True)[0].reshape(-1, self.d)
            return out, vis_feat
        else:
            pt_feat = x
            x = self.mlp(x)
            vis_feat = x.reshape(-1, n_pts, self.d)
            x = torch.max(x, -1, keepdim=True)[0].repeat(1,1,n_pts)
            return torch.concat([x, pt_feat], dim=1), vis_feat

class PointNetCls1024D(nn.Module):
    def __init__(self, k=2 ):
        super(PointNetCls1024D, self).__init__()
        self.k = k
        # self.mlp = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, k)
        # )
        
        # Conv1d hear is equivalent to fully connected layer
        self.mlp = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, k, 1)
        )
        
        self.pt_feat = PointNetfeat(global_feat=True)

    def forward(self, x):
        x = x.transpose(1, 2)
        x, vis_feature  = self.pt_feat(x)
        x = self.mlp(x[:,:,None]).reshape((-1,self.k))
        return F.log_softmax(x, dim=1), vis_feature # vis_feature only for visualization, your can use other ways to obtain the vis_feature


class PointNetCls256D(nn.Module):
    def __init__(self, k=2 ):
        super(PointNetCls256D, self).__init__()
        self.k = k
        self.pt_feat = PointNetfeat(global_feat=True, d=256)
        self.mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, k),
        )
    def forward(self, x):
        x = x.transpose(1, 2)
        x, vis_feature = self.pt_feat(x)
        x = self.mlp(x).reshape((-1,self.k))
        return F.log_softmax(x, dim=-1), vis_feature





class PointNetSeg(nn.Module):
    def __init__(self, k = 2):
        super(PointNetSeg, self).__init__()
        self.k = k
        self.pt_feat = PointNetfeat(global_feat=False,d=1024)
        self.mlp = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, k, 1)
        )


    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[1]

        x = x.transpose(1,2)
        x, _ = self.pt_feat(x)
        x = self.mlp(x).transpose(1,2).contiguous()

        return F.log_softmax(x, dim=-1)

