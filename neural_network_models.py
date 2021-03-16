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
