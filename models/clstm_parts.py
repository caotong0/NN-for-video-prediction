import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=(3, 3), bias=True):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------

        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Sequential(nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                            out_channels=4 * self.hidden_dim,
                                            kernel_size=self.kernel_size,
                                            padding=self.padding,
                                            bias=self.bias),
                                  nn.GroupNorm(4*self.hidden_dim//8, 4 * self.hidden_dim)
                                  # nn.BatchNorm2d(4*self.hidden_dim)
                                  )

    def forward(self, input_tensor, cur_state):
        # This part is more similar to LSTM
        # Sligthly different from the ConvLSTM
        # But it works fine
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f + 1)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvLayer, self).__init__()
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.gru = False
        if self.gru:
            self.convLSTM = ConvGRUCell(in_ch, out_ch)
        else:
            self.convLSTM = ConvLSTMCell(in_ch, out_ch)

    def forward(self, x=None, state=None):
        if state is None:
            state = self._init_hidden([x.size(0), self.out_ch, x.size(2), x.size(3)])
        if x is None:
            size_h = [state[0].size()[0], self.in_ch] + list(state[0].size()[2:])
            x = torch.zeros(size_h).cuda()
            # print(x.size())
        x = self.convLSTM(x, state)
        return x

    def _init_hidden(self, size):
        # h = torch.zeros(size).cuda()
        h, c = (torch.zeros(size).cuda(),
                torch.zeros(size).cuda())
        return h, c


class ConvGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=5):
        super(ConvGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.kernel_size = kernel_size
        # self.dropout = nn.Dropout(p=0.5)
        self.ConvGates = nn.Sequential(
            nn.Conv2d(self.input_size + self.hidden_size, 2 * self.hidden_size, self.kernel_size,
                      padding=kernel_size // 2),
            nn.GroupNorm(2 * self.hidden_size // 16, 2 * self.hidden_size))
        self.Conv_ct = nn.Sequential(nn.Conv2d(self.input_size + self.hidden_size, self.hidden_size, self.kernel_size,
                                               padding=kernel_size // 2),
                                     nn.GroupNorm(self.hidden_size // 16, self.hidden_size))
        dtype = torch.FloatTensor

    def forward(self, input, hidden):
        hidden = hidden[0]
        combined = torch.cat((input, hidden), 1)
        c1 = self.ConvGates(combined)
        (rt, ut) = c1.chunk(2, 1)
        # reset_gate = self.dropout(torch.sigmoid(rt))
        # update_gate = self.dropout(torch.sigmoid(ut))
        reset_gate = torch.sigmoid(rt)
        update_gate = torch.sigmoid(ut)
        gated_hidden = torch.mul(reset_gate, hidden)
        p1 = self.Conv_ct(torch.cat((input, gated_hidden), 1))
        ct = torch.tanh(p1)
        next_h = torch.mul(update_gate, hidden) + (1 - update_gate) * ct
        return next_h, next_h
