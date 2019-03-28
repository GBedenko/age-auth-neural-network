import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import csv


def plot_loss_function_results():

    x = []
    y = []

    with open('loss_function_results.csv', 'r') as csvfile:

        plots = csv.reader(csvfile, delimiter=',')

        for row in plots:
            x.append(float(row[0]))

    plt.plot(x, label='Loaded from loss_function_results.csv')
    plt.xlabel('Loss functions over time')
    plt.ylabel('Loss function output')
    plt.title('Loss Function Results over time')
    plt.legend()
    plt.show()


def accuracy_results():

    x = []
    y = []

    with open('accuracy_results.csv', 'r') as csvfile:

        plots = csv.reader(csvfile, delimiter=',')

        for row in plots:
            x.append(float(row[1]))

    plt.plot(x, label='Loaded from accuracy_results.csv')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Testing Accuracy Percentage')
    plt.title('Testing Accuracy over number of Epochs')
    plt.legend()
    plt.show()

plot_loss_function_results()