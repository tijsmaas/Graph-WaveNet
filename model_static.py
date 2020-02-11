import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys


# RNN model for every sensor
class StaticNet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
        super(StaticNet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.hidden = 32
        self.hidden_fc = 100
        self.receptive_field = 1

        self.b = torch.zeros(1, requires_grad=True).to(device)
        self.conv1 = nn.Conv2d(1, 1, kernel_size=1).to(device)

        self.supports = supports

        self.timeframes = out_dim + self.receptive_field


# input: [batch, vals, sensors, measurements]
    def forward(self, input):
        batch = input.size(0) # 64
        vals = input.size(1) # 2
        sensors = input.size(2) #207
        in_len = input.size(3)
        x = input

        # Take most recent measurement values
        all_sensors_input = x[:,0,:,-1]
        # Copy values of last timestep to all predictions [64, 207] -> [64, 1, 207, 12]
        static_val = all_sensors_input.view(batch, sensors, 1, 1)

        y = torch.arange(batch * self.timeframes * sensors).view(batch, sensors, self.timeframes, 1).cuda()
        a, b = torch.broadcast_tensors(static_val, y)

        # output.fill_(static_val * self.b)
        # [64, 5, 13, 1]
        output = a.transpose(1,2) + self.b
        # [64, 13, 5, 1]

        output = output[:, 1:, :, :]
        return output



    @classmethod
    def from_args(cls, args, device, supports, aptinit, **kwargs):
        defaults = dict(dropout=args.dropout, supports=supports,
                        addaptadj=args.addaptadj, aptinit=aptinit,
                        in_dim=args.in_dim, out_dim=args.seq_length,
                        residual_channels=args.nhid, dilation_channels=args.nhid)
        defaults.update(**kwargs)
        model = cls(device, args.num_nodes, **defaults)
        return model

