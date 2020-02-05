import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)


# RNN model for every sensor
class LSTMNet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
        super(LSTMNet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.hidden = 32
        self.hidden_fc = 100
        self.receptive_field = 1

        self.supports = supports
        self.rnn = [nn.LSTM(in_dim, self.hidden, layers, dropout=dropout).cuda() for _ in range(num_nodes)]
        self.rnn_all = nn.LSTM(in_dim * num_nodes, self.hidden, layers, dropout=dropout).cuda()
        self.fcn_all = torch.nn.Sequential(
            torch.nn.Linear(self.hidden, self.hidden_fc),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_fc, in_dim * num_nodes),
        ).cuda()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))


        self.end_conv_3 = nn.Conv2d(in_channels=residual_channels,
                                    out_channels=1,
                                    kernel_size=(1, 1),
                                    bias=True)


        self.timeframes = out_dim + self.receptive_field


# input: [batch, vals, sensors, measurements]
    def forward(self, input):
        in_len = input.size(3)
        if in_len<self.timeframes:
            x = nn.functional.pad(input,(self.timeframes-in_len,0,0,0))
        else:
            x = input
        # x = self.start_conv(x)
        skip = 0

        batch = 64
        vals = 2
        sensors = input.size(2) #207
        hidden = 32
        hidden_fc = 100
        layers = 2
        dropout = 0.3

        h0 = torch.randn(2, self.timeframes, self.hidden).cuda() # layers, .., hidden
        c0 = torch.randn(2, self.timeframes, self.hidden).cuda()

        # LSTM-FC torch.Size([64, 12, 414])
        all_sensors_input = x.flatten(1, 2).transpose(1, 2).cuda()
        # Run rnn for every timeseries step: [batch, timeseries, vals*sensors]
        output, (hn, cn) = self.rnn_all(all_sensors_input, (h0, c0))
        # FCN to convert global embedding to sensor values
        output = self.fcn_all(output)
        output = output.view(batch, self.timeframes, sensors, vals)
        # print('1', output.shape)
        output = output.transpose(1, 3)
        output = output[:, :, :, 1:]
        return output


        # for every sensor individually
        # c = torch.chunk(input, sensors, dim=2)
        # out_list = []
        # for idx, chunk in enumerate(c):
        #     single_sensor_input = chunk.squeeze()
        #     single_sensor_input_sw = single_sensor_input.transpose(1, 2)
        #     output, (hn, cn) = self.rnn[idx](single_sensor_input_sw, (h0, c0))
        #     out_list.append(output)
        # # input = torch.randn(64, 12, 2).to('cuda:0') # batch, .., input
        # output = torch.stack(out_list, dim=2)

        print(output.shape) # ([64, 13, 207, 32])

        output = output.transpose(1,3)
        pred_output = output[:, :, :, 1:]
        return self.end_conv_3(pred_output)

        # [batch, 32, sensors, 1 timestep]
        # x = F.relu(pred_output)
        # [batch, 512, sensors, 1 timestep]
        # x = F.relu(self.end_conv_3(x))





