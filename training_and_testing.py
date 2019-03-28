import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import imageio
import os
from AgeDataset import AgeDataset
from AgeCNN import AgeCNN
from argparse import ArgumentParser

def setup():

    # Instance of the training dataset
    training_dataset = AgeDataset(training=True)

    # Use pytorch DataLoader to be able to create batches for training
    training_data_loader = DataLoader(dataset=training_dataset,
                            batch_size=64,
                            shuffle=True,
                            num_workers=2)

    # Instance of the testing dataset
    testing_dataset = AgeDataset(training=False)

    # Use pytorch DataLoader to be able to create batches for training
    testing_data_loader = DataLoader(dataset=testing_dataset,
                            batch_size=64,
                            shuffle=True,
                            num_workers=2)

    # Instance of the convolutional neural network
    model = AgeCNN()

    # If an existing model available, load it
    if(os.path.isfile('latest_age_cnn_model')):
        model.load_state_dict(torch.load('latest_age_cnn_model'))
    
    # Optimizer which calculates the current gradient, only need to provide learning rate (SGD was used but other available which could be better)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    return training_data_loader, testing_data_loader, model, optimizer


# Function to do just the training of the network (now that we have test dataset as well and needs isolation)
def train(epoch, model, training_data_loader, optimizer):

    model.train() # Put the cnn model in train mode (pytorch knows we aren't predicting but are training)

    # An epoch is split into batches, which was decided by the data loader in pytorch, loops over each batch in an epoch
    for batch_idx, (data, target) in enumerate(training_data_loader):

        # Retrieves the pixel array data, and the target output array
        data, target = Variable(data), Variable(target)

        # Zeroes the optimiser
        optimizer.zero_grad()

        # Inputs the data into the model to get the current guess of the results array at this epoch for current input
        output = model(data)

        # Compute loss for this epoch, this case uses NLL Loss
        loss = F.nll_loss(output, target)

        # Runs back propagation to calculate inputs to get to this
        loss.backward()

        # Runs the optimiser one step to calculate the new gradient to be used
        optimizer.step()

        # Whenever batch index multiple of 10 (i.e. multiple of 640 because batch size is 64), print progress in this epoch, 
        # and print the current loss function result
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(training_data_loader.dataset),
                100. * batch_idx / len(training_data_loader), loss.item()))

    torch.save(model.state_dict(), 'latest_age_cnn_model')

# Function to just run the tests on a trained network (used in isolation with a training dataset)
def test(model, testing_data_loader):

    model.eval() # Sets model to test mode, lets pytorch knkow the model is being used to predict now, not to train

    test_loss = 0 # Will be populated with the total loss found for every test data matrix
    correct = 0 # Increases every time the highest output node matches the actual value

    # For each data matrix in test data loader class
    for data, target in testing_data_loader:

        # Data matrix and the target matrix for current test increment saved   
        data, target = Variable(data, volatile=True), Variable(target)

        # Run the data matrix/array of pixels through the model
        output = model(data)
        
        # Sum up the loss of NLL loss for every test in this epoch
        test_loss += F.nll_loss(output, target, size_average=False).item()

        # get the index of the max log-probability (node with highest value in output layer after running through model)
        pred = output.data.max(1, keepdim=True)[1]

        # If max output node is same as actual value then increment correct by 1
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    # Test loss becomes average, uses how many were tested
    test_loss /= len(testing_data_loader.dataset)

    # Output the accuracy of the results on all test data
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testing_data_loader.dataset),
        100. * correct / len(testing_data_loader.dataset)))


# Function to call the training and testing of the network, always using the latest model
def train_and_test_network(epochs):

    # Retrieve the datasets for pytorch and model
    training_data_loader, testing_data_loader, model, optimizer = setup()
    
    # For the input range
    for epoch in range(epochs):

        # Retrieve latest model if available
        if os.path.isfile('latest_age_cnn_model'):
            model.load_state_dict(torch.load('latest_age_cnn_model')) 

        # Train the retrieved model
        train(epoch, model, training_data_loader, optimizer)

        # Retrieve latest model if available
        if os.path.isfile('latest_age_cnn_model'):
            model.load_state_dict(torch.load('latest_age_cnn_model')) 

        # Test the most recent retrieved model
        test(model, testing_data_loader)

# If called on the command line call to run for number of epochs in args
if __name__ == "__main__":

    # Take user parameters of model directory and image path
    parser = ArgumentParser(add_help=True)
    parser.add_argument('--epochs', required=True, type=int)

    # Parse to global args object used in this file
    args = parser.parse_args()

    # Call predict_age with inputs and only print the result if running this file directly
    train_and_test_network(args.epochs)

