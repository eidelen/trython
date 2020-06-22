# from https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size, hidden_size, n_layers)
        self.l2o = nn.Linear(hidden_size, output_size)

    def forward(self, input, h, c):
        l, (h_new, c_new) = self.lstm(input, (h, c))
        output = self.l2o(l)
        return output, h_new, c_new

    def initHidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size), \
               torch.zeros(self.n_layers, 1, self.hidden_size)




def simple_test():
    net = RNN(10, 10, 1, 10)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # sample 1
    i00 = torch.zeros(1, 1, 10)
    i00[0,0,3] = 1.0
    i01 = torch.zeros(1, 1, 10)
    i01[0,0,6] = 1.0
    l0 = torch.ones(1, dtype=torch.long)
    l0[0] = 5

    # sample 2
    i10 = torch.zeros(1, 1, 10)
    i10[0,0,7] = 1.0
    i11 = torch.zeros(1, 1, 10)
    i11[0, 0, 1] = 1.0
    l1 = torch.ones(1, dtype=torch.long)
    l1[0] = 4

    optimizer.zero_grad()

    for i in range(1000):
        h, c = net.initHidden()

        # interchanging sample input
        if i % 2 == 0:
            i0, i1, l = i00, i01, l0
        else:
            i0, i1, l = i10, i11, l1

        # each sample has a sequence of length two
        _, h, c = net(i0, h, c)
        output, _, _ = net(i1, h, c)

        loss = criterion(torch.squeeze(output,0), l)
        loss.backward()
        optimizer.step()
        print(output)
        print(loss.item())


if __name__ == "__main__":
    # execute only if run as a script
    simple_test()