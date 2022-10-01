import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
# from camera import VideoCamera

# Some utilites
import numpy as np
from util import base64_to_pil

import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# Declare a flask app
app = Flask(__name__)



model_path = "./models/weights/vgg_model1.h5"
CATEGORIES = ['Acacia', 'Adenanthera microsperma', 'Adenium species', 'Anacardium occidentale', 'Annona squamosa', 'Artocarpus altilis', 'Artocarpus heterophyllus', 'Barringtonia acutangula']

model = tf.keras.models.load_model(model_path, compile=False)





def clf_predict(img):

    y_pred = model.predict(img)
    y_pred = y_pred.argmax(axis=-1)
    pred_class = CATEGORIES[y_pred[0]]

    return pred_class


# Home page
@app.route('/', methods=['POST', 'GET'])
def index():
    # Main page
    return render_template('index.html')



# VGG home page
@app.route('/vgg/',methods = ['POST', 'GET'])
def vgg():


    return render_template('vgg.html')


#Xception home page


@app.route('/xception/',methods = ['POST', 'GET'])
def resnet():
    

    return render_template('xception.html')


# # DensetNet home page
# @app.route('/densenet/',methods = ['POST', 'GET'])
# def densenet():
#
#
#     return render_template('densenet.html')


# Prediction page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)
        # Save the image to ./uploads
        img.save("./uploads/image.jpg")
        print("saved")

        img = cv2.imread("./uploads/image.jpg")
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
 


        img = img=cv2.resize(img,(224,224))
        img = img.reshape(-1, 224, 224, 3)
          
        class_lbl = clf_predict(img)
        print(class_lbl)
        pred_proba = 1
        # Serialize the result, you can add additional fields
        return jsonify(result=class_lbl, probability=pred_proba)

    return None



if __name__ == "__main__":
    app.run(debug=True)
  
