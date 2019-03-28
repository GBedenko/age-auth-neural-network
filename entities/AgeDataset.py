import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import imageio
import matplotlib.pyplot as plt

class AgeDataset(Dataset):
    """Represents our Diabetes dataset so that we can use it in pytorch."""

    # Initialize your data, download, etc.
    def __init__(self, training=True):

        datafile = './datasets/training_data_70k.csv' if(training) else './datasets/testing_data_30k.csv'

        # Import the data from file and split into inputs (the image filenames) and outputs (the age of person in the image)
        raw_data = np.loadtxt(datafile, delimiter=',', dtype=np.unicode)

        # Convert the ages values to floats (because it has to be saved as unicode for entire data)
        y_data = raw_data[:,-1]
        y_data = y_data.astype(np.long)
                
        self.len = raw_data.shape[0]
        self.raw_data = raw_data
        self.y_data = torch.from_numpy(y_data)


    def __getitem__(self, index):

        # Each image is in directory images followed by names stored in imported csv file (e.g. ./images/01/filename.jpg)
        image = imageio.imread('./images/' + self.raw_data[index, 0])

        new_tensor = torch.tensor(image)

        return new_tensor, self.y_data[index]


    def __len__(self):
        return self.len