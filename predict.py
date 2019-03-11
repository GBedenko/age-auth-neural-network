#!/usr/bin/env python

import logging
from argparse import ArgumentParser

import tensorflow as tf
from scipy.misc import imread
from tensorflow.contrib import predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tf.logging.set_verbosity(tf.logging.INFO)

def predict_age(model_dir, image_path):

    prediction_fn = predictor.from_saved_model(export_dir=model_dir, signature_def_key='serving_default')

    batch = []

    image = imread(image_path)
    output = prediction_fn({
        'image': [image]
    })
    
    return output["age_class"][0]


if __name__ == "__main__":
    predict_age('var/cnn-model/serving/1551650924/', '../303COM-Age-Verification-via-Facial-Recognition-App/photo.png')

