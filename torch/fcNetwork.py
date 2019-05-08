import torch
import torch.nn as nn

# learn if a number is positive or negative

class FullyConnected(nn.Module):

    def __init__(self):
        super(FullyConnected, self).__init__()
        self.h1 = nn.Linear(1,4)
        self.out = nn.Linear(4,2)

        self.s1 = nn.Sigmoid()
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.h1(x)
        x = self.s1(x)
        x = self.out(x)
        x = self.sm(x)
        return x


def testNet(net):
    n = 100;
    correct = 0
    wrong = 0

    for i in range(n):
        x = torch.randn(1,1)
        o = net(x)

        if x[0][0] > 0.0:
            if o[0][0] > o[0][1]:
                correct = correct + 1
            else:
                wrong = wrong + 1
        else:
            if o[0][0] < o[0][1]:
                correct = correct + 1
            else:
                wrong = wrong + 1

    assert wrong + correct == n

    print( correct / n * 100 )


myNet = FullyConnected()
params = list(myNet.parameters())
print(params[0])

for i in range(10000):
    x = torch.randn(1,1)
    y = torch.zeros(1,2)

    if x[0][0] > 0.0:
        y[0][0] = 1.0
    else:
        y[0][1] = 1.0

    out = myNet(x)

    crit = nn.MSELoss()
    errVal = crit(out,y)

    myNet.zero_grad()
    errVal.backward()

    learning_rate = 0.01
    for f in myNet.parameters():
        f.data.sub_(f.grad.data * learning_rate)

    if i % 25 == 0:
        testNet(myNet)






