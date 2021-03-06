import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.tgcn import *
from model.utils.graph import Graph

class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)  #缓存区,可通过A访问数据

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn =  nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            # st_gcn(128, 256, kernel_size, 2, **kwargs),
            # st_gcn(256, 256, kernel_size, 1, **kwargs),
            # st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))

        self.st_cnn_networks = nn.ModuleList((
            st_cnn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_cnn(64, 64, kernel_size, 1, **kwargs),
            st_cnn(64, 64, kernel_size, 1, **kwargs),
            st_cnn(64, 64, kernel_size, 1, **kwargs),
            st_cnn(64, 128, kernel_size, 2, **kwargs),
            st_cnn(128, 128, kernel_size, 1, **kwargs),
            st_cnn(128, 128, kernel_size, 1, **kwargs),
            # st_cnn(128, 256, kernel_size, 2, **kwargs),
            # st_cnn(256, 256, kernel_size, 1, **kwargs),
            # st_cnn(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
        # CNN network

        #self.cnn1 = nn.Conv2d(in_channels,in_channels,  kernel_size=1)
        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)
        #self.fcn = nn.Linear(256, num_class)
    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size() #(64,3,300,18,1)
        # permute()将tensor的维度换位
        x = x.permute(0, 4, 3, 1, 2).contiguous() #(64,1,18,3,300)
        x = x.view(N * M, V * C, T) #(64,54,300)
        x = self.data_bn(x) # 将某一个节点的（X,Y,C）中每一个数值在时间维度上分别进行归一化
        x = x.view(N, M, V, C, T) #(64,1,18,3,300)
        x = x.permute(0, 1, 3, 4, 2).contiguous() #（64,1,3,300,18）
        x = x.view(N * M, C, T, V) #（64,3,300,18）
        x_cnn = x
        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        for gcnn in self.st_cnn_networks:
            x_cnn = gcnn(x_cnn)

        # global pooling
        # 此处的x是运行完所有的卷积层之后在进行平均池化之后的维度x=(64,256,1)
        x = torch.cat((x,x_cnn),1)
        #shape = x.shape
        x = F.avg_pool2d(x, x.size()[2:]) # pool层的大小是(300,18)
        # （64,256,1,1）
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        # （64,400）
        x = x.view(x.size(0), -1)
        return x

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance) #（64,256,300,18）

        _, c, t, v = x.size() #（64,256,300,18）
        # feature的维度是（64,256,300,18,1）
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        # (64,400,300,18)
        x = self.fcn(x)
        # output: (64,400,300,18,1)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature
class st_cnn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = SpatialConv(in_channels, out_channels,
                                         kernel_size[1])
        # self.tcn()没有改变变量x.size()
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x) + res #(64,64,300,18)
        # (64,64,300,18)
        return self.relu(x)
class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])
        # self.tcn()没有改变变量x.size()
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res #(64,64,300,18)
        # (64,64,300,18)
        return self.relu(x), A


if __name__ == '__main__':
    graph_arg = dict(layout= 'openpose',strategy= 'spatial') #{layout: 'openpose',strategy: 'spatial'}
    model = Model(3,400,graph_args=graph_arg,edge_importance_weighting=True)
    data =  torch.FloatTensor(64,3,60,18,1)
    model = model.cuda()

    data = data.cuda()
    result,shape = model(data)
    print("size: ",result.size())
    print("shape: ", shape)

    print(model)