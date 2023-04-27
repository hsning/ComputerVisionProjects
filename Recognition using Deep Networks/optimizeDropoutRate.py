# Hao Sheng (Jack) Ning Recognition using Deep Networks
# Used for task 4
# CS 5330 Computer Vision
# Spring 2023
# import statements
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Initialize variables
n_epochs = 10
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
train_losses = []
test_losses = []
train_counter = []


random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(42)

# Initialize data loaders
train_loader = torch.utils.data.DataLoader(
  datasets.MNIST('data', train=True, download=True,
                             transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  datasets.MNIST('data', train=False, download=True,
                             transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)
test_counter = [i*len(train_loader.dataset) for i in range(1,n_epochs+1)]
examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)
# class definitions
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(5,5))
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(5,5))
        self.conv2_drop = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=320,out_features=50)
        self.fc2 = nn.Linear(in_features=50,out_features=10)

#     # computes a forward pass for the network
#     # methods need a summary comment
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x = x.view(-1, self.conv2.out_channels*16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x,dim=1)
        return x
    
# useful functions with a comment for each function
def train( loader,model,loss_function, optimizer,epoch ):
    size = len(loader.dataset)
    model.train()
    for batch, (X, y) in enumerate(loader):
        X, y = X.to("cpu"), y.to("cpu")

        # Compute prediction error
        pred = model(X)
        loss = loss_function(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            #2print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            train_losses.append(loss)
            train_counter.append(
            (batch*64) + ((epoch-1)*len(train_loader.dataset)))
    return

def test(dataloader, model, loss_fn, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to("cpu"), y.to("cpu")
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    test_losses.append(test_loss)
    if(epoch==n_epochs):
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# main function (yes, it needs a comment too)
def main(argv):
    dropoutRates = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    # handle any command line arguments in argv
    print("hello world")
    print(example_data.shape[0])
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    fig
    plt.show()
    
    loss_fn = nn.CrossEntropyLoss()
    
    
    for j in range(len(dropoutRates)):
        modelA = MyNetwork().to("cpu")
        modelA.conv2_drop=nn.Dropout(p=dropoutRates[j])
        print("With drop out rate of: "+str(dropoutRates[j]))
        optimizer = torch.optim.SGD(modelA.parameters(), lr=1e-3)
        for t in range(1, n_epochs + 1):
            #print(f"Epoch {t}\n-------------------------------")
            #test_counter.append(t*len(train_loader.dataset))
            train(train_loader, modelA, loss_fn, optimizer,t)
            test(test_loader, modelA, loss_fn,t)
            #print("Done!")
    # main function code
    return

if __name__ == "__main__":
    main(sys.argv)