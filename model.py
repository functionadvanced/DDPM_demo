import torch
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, T):
        # the structure is convolutional layer, 3 linear layers, and a skip connection from the input to the output
        super(Model, self).__init__()
        output_size = input_size
        # embedding layer, embedding of t
        self.embedding = nn.Embedding(T, embedding_size)
        # convolutional layer, input size is the sum of input_size and embedding_size
        # number of output channels
        num_channels = 30
        self.conv = nn.Conv1d(1, num_channels, 3)
        # size of the output of the convolutional layer
        self.conv_size = (input_size - 2) * num_channels
        # 3 linear layers
        self.fc1 = nn.Linear(self.conv_size + embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        # skip connection from the input to the output
        self.skip = nn.Linear(input_size, output_size)

    def forward(self, x_input, t):
        t = self.embedding(t)
        # convolutional layer
        x = F.relu(self.conv(x_input.unsqueeze(1)).squeeze(1))
        # flatten the output of the convolutional layer
        x = x.view(-1, self.conv_size)
        # concatenate x and t
        x = torch.cat([x, t], dim=1)
        # 2 linear layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # skip connection from the input to the output
        x = x + self.skip(x_input)
        return x