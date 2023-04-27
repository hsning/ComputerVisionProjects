# Hao Sheng (Jack) Ning Recognition using Deep Networks
# Used for task #3
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
n_epochs = 150
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
        self.mul_factor=2
        self.fc1 = nn.Linear(in_features=320,out_features=50)
        self.fc2 = nn.Linear(in_features=50,out_features=10)

#     # computes a forward pass for the network
#     # methods need a summary comment
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), self.mul_factor))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)
    
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
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            train_losses.append(loss)
            train_counter.append(
            (batch*64) + ((epoch-1)*len(loader.dataset)))
    return

def test(dataloader, model, loss_fn):
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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# greek data set transform
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = transforms.functional.rgb_to_grayscale( x )
        x = transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = transforms.functional.center_crop( x, (28, 28) )
        return transforms.functional.invert( x )

# DataLoader for the Greek data set
greek_train = torch.utils.data.DataLoader(
    datasets.ImageFolder( "greek_train-3",
                          transform = transforms.Compose( [transforms.ToTensor(),
                          GreekTransform(),
                          transforms.Normalize(
                          (0.1307,), (0.3081,) ) ] ) ),
        batch_size = 5,
        shuffle = True )

# main function (yes, it needs a comment too)
def main(argv):
    # main function code
    labels=["alpha","beta","gamma"]
    model = MyNetwork().to("cpu")
    model.load_state_dict(torch.load("result/model.pth"))
    loss_fn = nn.CrossEntropyLoss()
    for param in model.parameters():
        param.requires_grad = False
    model.fc2=nn.Linear(in_features=50,out_features=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    print(model)
    fig = plt.figure()
    trainingExamples = enumerate(greek_train)
    batch_idx, (trainingExamples_data, trainingLabels) = next(trainingExamples)
    for i in range(1,4):
        plt.subplot(3,1,i)
        plt.tight_layout()
        plt.imshow(trainingExamples_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(labels[trainingLabels[i]]))
        plt.xticks([])
        plt.yticks([])
    fig
    plt.show()
    for t in range(1, n_epochs + 1):
      print(f"Epoch {t}\n-------------------------------")
      #test_counter.append(t*len(train_loader.dataset))
      train(greek_train, model, loss_fn, optimizer,t)
      print("Done!")
    #data_transform = transforms.Compose([transforms.RandomInvert(1),transforms.Resize((28,28)),transforms.Grayscale(num_output_channels=1),
                                     #transforms.ToTensor()])
    plt.plot(train_counter, train_losses, color='blue')
    plt.legend(['Train Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    fig
    plt.show()
    test_data = datasets.ImageFolder('testDataGreek-3/', transform = transforms.Compose( [transforms.ToTensor(),                                                             
                          GreekTransform(),
                          transforms.Normalize(
                          (0.1307,), (0.3081,) ) ] ))
    myTest_loader = torch.utils.data.DataLoader(test_data,batch_size=10, shuffle=True)
    myExamples = enumerate(myTest_loader)
    batch_idx, (myExample_data, _) = next(myExamples)
    with torch.no_grad():
        test_output = model(myExample_data)
    fig = plt.figure()
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.tight_layout()
        plt.imshow(myExample_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(
            labels[test_output.data.max(1, keepdim=True)[1][i].item()]))
        plt.xticks([])
        plt.yticks([])
    fig
    plt.show()
    #batch_idx, (example_test_data, _) = next(myTest_loader)
    model.eval()
    return

if __name__ == "__main__":
    main(sys.argv)