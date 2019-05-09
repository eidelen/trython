import torch
import torch.nn as nn
import torch.optim as optim

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



def getLabeledInput(nbrSamples):

    # input
    x = torch.randn(nbrSamples, 1)

    # output -> positive or negative
    y = torch.zeros(nbrSamples, 2)
    for l in range(nbrSamples):
        if x[l][0] > 0.0:
            y[l][0] = 1.0
        else:
            y[l][1] = 1.0

    return x, y



def testNet(net):
    n = 100;
    correct = 0
    x,y = getLabeledInput(n)
    o = net(x)

    for i in range(n):
        if x[i][0] > 0.0:
            if o[i][0] > o[i][1]:
                correct = correct + 1
        else:
            if o[i][0] < o[i][1]:
                correct = correct + 1

    return correct / n



print("pyTorch version" + torch.__version__)

print("\n\nDo own optimisation")
myNet = FullyConnected()

optimizer = optim.SGD(myNet.parameters(), lr=0.01)
crit = nn.MSELoss()

for i in range(100000):

    batchSize = 25

    x,y = getLabeledInput(batchSize)

    # feedforward and compute error
    out = myNet(x)
    errVal = crit(out,y)

    # compute gradient
    optimizer.zero_grad()
    errVal.backward()

    # adapt bias and weights
    optimizer.step()

    # test performance
    if i % 50 == 0:
        success = testNet(myNet)
        print(success*100.0)
        if success > 0.99:
            print("Well done...")
            break


