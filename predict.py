#!/usr/bin/env python

# Import argparse so that file paths can be entered on commandline
from argparse import ArgumentParser

# Import Tensorflow to understand model path
import tensorflow as tf
from tensorflow.contrib import predictor

# Import scipy to convert input image to array
from scipy.misc import imread

def predict_age(model_dir, image_path):
    """Function which takes the directory of the CNN model and an image as input
       Creates a tensorflow prediction function based on the latest model
       Runs the function for the provided image and returns the predicted age of the person in the photo"""

    # Constructs a tensorflow predictor from the input model
    prediction_fn = predictor.from_saved_model(export_dir=model_dir, signature_def_key='serving_default')

    # Read input image as an array using scipy
    image = imread(image_path)

    # Use the tensorflow function created from the cnn model with the input image
    output = prediction_fn({
        'image': [image]
    })
    
    # From tensorflow functions's output, return the age of the person in the image
    return output["age_class"][0]


# If called on the command line requires args for model and image
# Instead the predict_age function can be called from another module
if __name__ == "__main__":

    # Take user parameters of model directory and image path
    parser = ArgumentParser(add_help=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--image-path', required=True)

    # Parse to global args object used in this file
    args = parser.parse_args()

    # Call predict_age with inputs and only print the result if running this file directly
    print(predict_age(args.model_dir, args.image_path))

