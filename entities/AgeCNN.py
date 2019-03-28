import torch
import torch.nn as nn
import torch.nn.functional as F

class AgeCNN(nn.Module):
    """
    Representing a convolutional neural network which uses 2 cnn layers, a max pooling layer, and then a linear layer
    Input is 1 input signal for the input pixels
    Output is 10 nodes the the different digits possible (0-9)
    """

    def __init__(self):

        super(AgeCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 10, kernel_size=5) # Input of 3 for RGB images
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # Another CNN layer to hopefully improve accuracy
        self.conv3 = nn.Conv2d(20, 40, kernel_size=5) # Another CNN layer to hopefully improve accuracy
        self.conv4 = nn.Conv2d(40, 80, kernel_size=5) # Another CNN layer to hopefully improve accuracy
        self.mp = nn.MaxPool2d(2) # Max pooling of size 2x2 pixel grids
        self.fc = nn.Linear(8000, 100) # Linear function to final output layer for age range
        # Input to linear function is found from matrix mismatch, will change everytime new layers added

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = x.type('torch.FloatTensor')

        x = F.relu(self.mp(self.conv1(x))) # First layer is cnn, then max pooling, then relu to reduce (like sigmoid)
        x = F.relu(self.mp(self.conv2(x)))
        x = F.relu(self.mp(self.conv3(x)))
        x = F.relu(self.mp(self.conv4(x)))
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = self.fc(x)
        return F.log_softmax(x)
        