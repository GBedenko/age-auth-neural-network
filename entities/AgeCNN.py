import torch
import torch.nn as nn
import torch.nn.functional as F

class AgeCNN(nn.Module):
    """
    Representing a Convolutional Neural Network which uses cnn layers, a max pooling layer, and then a linear layer
    Input is 3 input signals for the input pixels (RGB images)
    Output is 100 nodes the the different ages from 0-99 years old
    """

    def __init__(self):

        super(AgeCNN, self).__init__()

        # Convolutional Layers apply a convole filter - output node size can be adjusted to hopefully help change accuracy
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5) # Input of 3 for RGB images
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # Another CNN layer to hopefully improve accuracy
        self.conv3 = nn.Conv2d(20, 40, kernel_size=5) # Another CNN layer to hopefully improve accuracy
        self.conv4 = nn.Conv2d(40, 80, kernel_size=5) # Another CNN layer to hopefully improve accuracy

        # Max pooling layer reduces every 2x2 pixel grid to the highest value in each grid for the output
        self.mp = nn.MaxPool2d(2) # Max pooling of size 2x2 pixel grids

        # Linear function to define how many output nodes there is in the final layer - using 100 for ages, could be reduced
        self.fc = nn.Linear(8000, 100) # Linear function to final output layer for age range
        # Input to linear function is found from matrix mismatch, will change everytime new layers added

    def forward(self, x):

        # PyTorch tensor needs reordering so that the 3D image and batch size can be processed by the model
        x = x.permute(0, 3, 2, 1)
        
        # Ensure all values in tensor are float so the model can use them
        x = x.type('torch.FloatTensor')

        # Layers use cnn filters, then max pooling, then relu to reduce to 0-1 value, repeated for each convolve filter
        x = F.relu(self.mp(self.conv1(x))) 
        x = F.relu(self.mp(self.conv2(x)))
        x = F.relu(self.mp(self.conv3(x)))
        x = F.relu(self.mp(self.conv4(x)))
        
        x = x.view(x.size(0), -1)  # flatten the tensor

        # Run through final linear layer
        x = self.fc(x)
        
        return F.log_softmax(x)
        