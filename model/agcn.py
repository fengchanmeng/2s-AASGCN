import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)



class ChannelAttentionModule(nn.Module):
    def __init__(self,channel,ratio=16):
        super(ChannelAttentionModule,self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.max_pool=nn.AdaptiveAvgPool2d(1)

        self.shared_MLP=nn.Sequential(
            nn.Conv2d(channel, channel//ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//ratio,channel, 1, bias=False)

        )
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        avgout=self.shared_MLP(self.avg_pool(x))
        maxout=self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout+maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule,self).__init__()
        self.conv2d=nn.Conv2d(in_channels=2,out_channels=1,kernel_size=7,stride=1,padding=3)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        avgout=torch.mean(x,dim=1,keepdim=True)
        maxout, _=torch.max(x,dim=1,keepdim=True)
        out=torch.cat([avgout,maxout],dim=1)
        out=self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self,channel):
        super(CBAM,self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self,x):
        out=self.channel_attention(x)*x
        out=self.spatial_attention(out)*out
        return out

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)#padding保证卷积之前和之后输入输出维度不变
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()#非线性激活函数
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x



class unit_gcn(nn.Module):#spatial GCN+bn+relu
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=5):
        super(unit_gcn, self).__init__()
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))#自适应机制
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset
        
        self.in_channels=in_channels
        self.out_channels=out_channels
        
        self.conv_list=nn.ModuleList([
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
                padding=0,
                stride=1
            ) for i in range(self.num_subset)
        ])

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

 
        bn_init(self.bn, 1e-6)

        for conv in self.conv_list:
            conv_init(conv)
        
        self.cbam = CBAM(out_channels)

    def forward(self, x):
        N, C, T, V = x.size()

        A = self.A.cuda(x.get_device())
        A = A + self.PA

        y = None
        for i,a in enumerate(A):
            xa = x.view(-1,V).mm(a).view(N,C,T,V)

            if i == 0:
                y = self.conv_list[i](xa)
            else:
                y=y+self.conv_list[i](xa)
        y = self.cbam(y)
        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)



class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)#对小批量(mini-batch)的2d或3d输入进行批标准化(Batch Normalization)操作。

        self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 64, A)
        self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A)
        self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A)
        self.l10 = TCN_GCN_unit(256, 256, A)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)#permute转换成(64,2,25,3,300).view(64,2*25*3,300)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x)
