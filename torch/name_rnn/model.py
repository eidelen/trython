# from https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size )
        self.i2a = nn.Linear(input_size + hidden_size, output_size * 2 + hidden_size * 2)
        self.a2b = nn.Linear(output_size * 2 + hidden_size * 2, output_size * 4)
        self.b2o = nn.Linear(output_size * 4, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1) # concatenate the two vectors in direction 1
        a = self.i2a(combined)
        hidden = self.i2h(combined)
        b = self.a2b(a)
        output = self.b2o(b)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))
