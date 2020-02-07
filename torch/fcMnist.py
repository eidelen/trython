# adapted https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import time


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 1 input image channel, 12 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



class FullyConnected(nn.Module):

    def __init__(self):
        super(FullyConnected, self).__init__()
        self.h1 = nn.Linear(784,30)
        self.out = nn.Linear(30,10)

        self.s1 = nn.Sigmoid()
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # reshape 2d image to vector
        x = self.h1(x)
        x = self.s1(x)
        x = self.out(x)
        x = self.sm(x)
        return x


# show grayscale image
def dispImage(img):
    img = img / 2.0 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(npimg[0], cmap='gray', vmin=0, vmax=1)
    plt.show()


def doLabelMatrix(l):
    n = l.shape[0]
    mat = torch.zeros(n, 10)
    for i in range(n):
        label = l[i]
        mat[i,label] = 1.0

    return mat;



def showMultipleImages(imgs):
    plt.close('all')
    n = min( 144, len(imgs) )
    fig = plt.figure(figsize=(12, 12))
    for i in range(1, n+1):
        img = imgs[i - 1] / 2.0 + 0.5
        npimg = img.numpy()
        fig.add_subplot(12, 12, i)
        plt.imshow(npimg[0], cmap='gray', vmin=0, vmax=1)

    plt.show(block=False)
    plt.pause(1)



print("pyTorch version" + torch.__version__)
print( 'Cuda is' + (' on' if torch.cuda.is_available() else ' off'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Use computational device: %s" % device)

# load data
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5, ))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,
                                          shuffle=True, num_workers=0)
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10,
                                         shuffle=False, num_workers=0)


# training
net = ConvNet()
net.to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

for epoch in range(10000):

    running_loss = 0.0
    wrongImgList = []
    start = time.time()

    for i, batch in enumerate(trainloader,0):
        x, y = batch

        y = doLabelMatrix(y).to(device)

        optimizer.zero_grad()
        out = net(x.to(device))

        loss = criterion(out,y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:
            print('[%d, %5d] loss: %.8f' % (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0


    # test with testdata
    correct = 0
    total = 0
    wrongImgList.clear()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            out = net(images.to(device))
            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            corrList = (predicted == labels.to(device))
            correct += corrList.sum().item()
            # copy wrong detected
            for c in range(corrList.shape[0]):
                if not corrList[c]:
                    wrongImgList.append(images[c])

    runningtime = time.time() - start
    print('Accuracy of the network on the 10000 test images: %.2f  (%d,  %d)%%   -  time: %.1f ' % (
            100 * correct / total , correct, total, runningtime))

    # showMultipleImages(wrongImgList)

