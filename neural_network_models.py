import torch
import torch.nn as nn

# define network arc.
# ================================================================
input_size = 784
output_size = 10
hidden1_size = 50
hidden2_size = 50


# ================================================================
# define neural network class.
# ================================================================
class Net(nn.Module):
    def __init__(self, weights1=torch.ones(hidden1_size, input_size), bias1=torch.ones(hidden1_size),
                 weights2=torch.ones(hidden1_size, hidden2_size), bias2=torch.ones(hidden2_size),
                 weights3=torch.ones(output_size, hidden2_size), bias3=torch.ones(output_size)):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(weights1.shape[1], weights1.shape[0])
        with torch.no_grad():
            self.fc1.weight.copy_(weights1)
            self.fc1.bias.copy_(bias1)

        self.fc2 = nn.Linear(weights2.shape[1], weights2.shape[0])
        with torch.no_grad():
            self.fc2.weight.copy_(weights2)
            self.fc2.bias.copy_(bias2)

        self.fc3 = nn.Linear(weights3.shape[1], weights3.shape[0])
        with torch.no_grad():
            self.fc3.weight.copy_(weights3)
            self.fc3.bias.copy_(bias3)

    def forward(self, x):
        x = x.reshape(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return torch.log_softmax(x, dim=-1)


class ConvNet(nn.Module):
    def __init__(self, weights1=torch.ones(16, 1, 4, 4), weights2=torch.ones(32, 16, 4, 4),
                 weights3=torch.ones(800, 100), weights4=torch.ones(100, 10)):
        super(ConvNet, self).__init__()

        self.layer1 = nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=0)
        with torch.no_grad():
            self.layer1.weight.copy_(weights1)
            self.layer1.bias.copy_(torch.zeros(16))

        self.relu1 = nn.ReLU()

        self.layer2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0)
        with torch.no_grad():
            self.layer2.weight.copy_(weights2)
            self.layer2.bias.copy_(torch.zeros(32))

        self.relu2 = nn.ReLU()

        self.fc1 = nn.Linear(weights3.shape[1], weights3.shape[0])
        print("weights3.shape[1]=",weights3.shape[1],"weights3.shape[0],=",weights3.shape[0])
        with torch.no_grad():
            self.fc1.weight.copy_(weights3)
            self.fc1.bias.copy_(torch.zeros(100))

        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(weights4.shape[1], weights4.shape[0])
        with torch.no_grad():
            self.fc2.weight.copy_(weights4)
            self.fc2.bias.copy_(torch.zeros(10))

    def forward(self, x):

        out = self.layer1(x)
        out = self.relu1(out)
        out = self.layer2(out)

        out = self.relu2(out)
  

        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.fc2(out)
        return torch.log_softmax(out, dim=-1)
