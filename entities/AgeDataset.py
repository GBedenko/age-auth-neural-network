import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import imageio
import matplotlib.pyplot as plt

class AgeDataset(Dataset):
    """Represents the IMDB dataset of images and the age of the person in each image
       This class is needed so that it can be used in pytorch."""

    # Initialize your data, download, etc.
    def __init__(self, training=True):

        # Depending on if training or test dataset is being initialised, retrieve data from correct file
        datafile = './datasets/training_data_70k.csv' if(training) else './datasets/testing_data_30k.csv'

        # Import the data from file and split into inputs (the image filenames) and outputs (the age of person in the image)
        raw_data = np.loadtxt(datafile, delimiter=',', dtype=np.unicode)
        
        y_data = raw_data[:,-1] # Outputs (y_data) is the age column of the dataset, numpy array of final column
        y_data = y_data.astype(np.long) # Convert the ages values to floats (because it has to be saved as unicode for entire data)
                
        self.len = raw_data.shape[0] # Saves the number of records in this instance of dataset
        self.raw_data = raw_data # Initialise object with all data from the dataset
        self.y_data = torch.from_numpy(y_data) # Initialise the outputs as a tensor object of the ages column


    def __getitem__(self, index):

        # Each image is in directory images followed by names stored in imported csv file (e.g. ./images/01/filename.jpg)
        image = imageio.imread('./images/' + self.raw_data[index, 0])

        # Create tensor of the image numpy array, so that pytorch can use it
        new_tensor = torch.tensor(image)

        # Return the image tensor input and the age value output
        return new_tensor, self.y_data[index]


    def __len__(self):
        
        # The number of records in this instance's dataset
        return self.len