from models.clstm_parts import *
import torch.nn as nn
import torch


class StaBriNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(StaBriNet, self).__init__()
        d = [n_channels, 32, 64, 128, 128, 256, 1]
        self.d = d
        self.layers = 5
        self.net = torch.nn.ModuleList()
        for i in range(self.layers):
            self.net.append(ConvLayer(d[i], d[i + 1]))
        for i in range(self.layers):
            self.net.append(ConvLayer(d[i], d[i + 1]))

        self.flowconv = nn.Sequential(nn.Conv2d(sum(d[1:self.layers + 1]), sum(d[1:self.layers + 1]), 1),
                                      nn.GroupNorm(4, sum(d[1:self.layers + 1])))
        self.outconv = nn.Sequential(nn.Conv2d(sum(d[1:self.layers + 1]), n_classes, 1),
                                     nn.GroupNorm(1, n_classes))
        self.relu = nn.ReLU()

    def forward(self, input, numpre=10, numin=5):
        output = []
        hidden = [None] * self.layers
        #   Encoder
        for i in range(numin):
            # B T C W H  turns into   B C W H
            x = input[:, i]
            collect_h = []
            for j in range(self.layers):
                hidden[j] = self.net[j](x, hidden[j])

                x = hidden[j][0]
                collect_h.append(x)

        x = torch.cat([hidden[j][0] for j in range(self.layers)], dim=1)
        flow = self.relu(self.flowconv(x))
        # Decoder
        x = input[:, numin - 1]
        for i in range(numpre):
            collect_h = []
            left = 0
            for j in range(self.layers):
                x = x + flow[:, left:left + self.d[j]]
                left += self.d[j]
                hidden[j] = self.net[j + self.layers](x, hidden[j])
                x = hidden[j][0]
                collect_h.append(x)

            x = torch.cat(collect_h, dim=1)
            flow = self.relu(self.flowconv(x))
            x = torch.sigmoid(self.outconv(flow))
            output.append(x)

        return torch.stack(output, dim=1)

class clstm(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(clstm, self).__init__()
        d = [n_channels, 32, 64, 128, 128, 256, 1]
        self.h = 5
        self.net = torch.nn.ModuleList()
        for i in range(self.h):
            self.net.append(ConvLayer(d[i], d[i + 1]))

        for i in range(self.h):
            self.net.append(ConvLayer(d[i], d[i + 1]))

        self.conv = nn.Sequential(nn.Conv2d(sum(d[1:self.h + 1]), n_classes, 1)
                                  , nn.GroupNorm(1, n_classes)
                                  )
        self.relu1 = nn.ReLU()

    def forward(self, input, numpre=10, numin=5):
        output = []
        hidden = [None] * self.h

        #   Encoder
        for i in range(numin):
            # B T C W H  turns to   B C W H
            x = input[:, i]
            for j in range(self.h):
                hidden[j] = self.net[j](x, hidden[j])
                x = hidden[j][0]

        # Decoder
        x = None
        # s shape state flow x
        for i in range(numpre):
            collect_h = []
            for j in range(self.h):
                hidden[j] = self.net[j](x, hidden[j])
                x = hidden[j][0]
                collect_h.append(x)

            x = torch.cat(collect_h, dim=1)
            x = torch.sigmoid(self.conv(x))
            output.append(x)

        return torch.stack(output, dim=1)

