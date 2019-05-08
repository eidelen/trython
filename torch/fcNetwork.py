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
    x = torch.randn(n,1)
    o = net(x)

    for i in range(n):
        if x[i][0] > 0.0:
            if o[i][0] > o[i][1]:
                correct = correct + 1
        else:
            if o[i][0] < o[i][1]:
                correct = correct + 1

    return correct / n




myNet = FullyConnected()
params = list(myNet.parameters())
print(params[0])

for i in range(100000):

    batchSize = 25

    # create random input batch
    x = torch.randn(batchSize,1)

    # lable the input
    y = torch.zeros(batchSize,2)
    for l in range(batchSize):
        if x[l][0] > 0.0:
            y[l][0] = 1.0
        else:
            y[l][1] = 1.0

    # feedforward and compute error
    out = myNet(x)
    crit = nn.MSELoss()
    errVal = crit(out,y)

    # compute gradient
    myNet.zero_grad()
    errVal.backward()

    # adapt bias and weights
    learning_rate = 0.01
    for f in myNet.parameters():
        f.data.sub_(f.grad.data * learning_rate)

    # test performance
    if i % 50 == 0:
        success = testNet(myNet)
        print(success*100.0)
        if success > 0.99:
            print("Well done...")
            break

