#!/usr/bin/env python

# Import Flask to host server
from flask import Flask, request, render_template

# Import modules for functionality
import predict
import json

# Create a Flask app to run as a web server to host local API
app = Flask(__name__)


# GET request for predicted age retrieval
@app.route("/determine_age")
def determine_age():

    predicted_age = predict.predict_age('var/cnn-model/serving/1551650924/', '../303COM-Age-Verification-via-Facial-Recognition-App/photo.png')

    # Save scan result as a dictionary
    age_data = {"predicted_age": predicted_age}

    # Convert to a json object age_data will be returned
    age_json = json.dumps(age_data)

    # Return int result as json object
    return age_json


if __name__ == "__main__":
    # Run app on localhost:8081 for testing purposes
    app.run(debug=True, host="localhost", port=8081)

