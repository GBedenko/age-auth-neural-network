#!/usr/bin/env python3

# Import Flask to host server
import flask
from flask import Flask, request, render_template

# Import modules for functionality
from predict import predict_age
import json

# Create a Flask app to run as a web server to host local API
app = Flask(__name__)


# GET request for predicted age retrieval
@app.route("/determine_age")
def determine_age():

    # Use the exported CNN model and the image captured by the other API
    predicted_age = predict_age('latest_age_cnn_model.pth', '../age-auth/photo.png')

    # Save scan result as a dictionary
    age_data = {"predicted_age": predicted_age}

    # Convert to a json object age_data will be returned
    age_json = json.dumps(age_data)

    # Return int result as json object
    return age_json


if __name__ == "__main__":
    
    # Run app on localhost:8081 for testing purposes
    app.run(debug=True, host="localhost", port=8081)

