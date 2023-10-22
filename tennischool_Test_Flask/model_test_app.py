from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
#from some_module import extract_coordinates

flask_app = Flask(__name__)
model = pickle.load(open(r"C:\Users\rch\Desktop\대학관련\tennischool\extract_coordinates_function.pkl", "rb"))


@flask_app.route("/")

def Home():
    return render_template("index.html")


@flask_app.route("/predict", methods = ["POST"])

def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    return render_template("index.html", prediction_text = "비슷한 선수는....{}".format(prediction))


if __name__ == "__main__":
    flask_app.run(debug = True)