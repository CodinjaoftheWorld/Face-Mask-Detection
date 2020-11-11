import sys
import os
import glob
import re
import numpy as np
import cv2
import requests
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


# Load the face detection model of mobilenet_v2
prototxtPath = os.path.sep.join(["./face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(["./face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)


# Load the mask detection trained model
model = load_model('mask_detector.model') 
model._make_predict_function()    
conf = 0.5



# Define the flask app
app = Flask(__name__)


# define a function to predict the model
def model_predict(img_path, model, net):

	# Load image
	#img = image.load_img(img_path)
	img = cv2.imread(img_path)
	(h,w) = img.shape[ :2]

	#construct a blob from the image
	blob = cv2.dnn.blobFromImage(img, 1.0, (300,300), (104.0, 177.0, 123.0))

	# Pass the blob through the network and obtain the face detections
	net.setInput(blob)
	detections = net.forward()


	# Loop through the detections
	for i in range(0, detections.shape[2]):
		confidence = detections[0,0,i,2]

		if confidence > conf:
			box = detections[0,0,i,3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))


			# load the input image (224x224) and preprocess it
			face = img[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224,224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			(mask, withoutMask) = model.predict(face)[0]

			if mask > withoutMask:
				return "Mask"
			else:
				return "No Mask"




@app.route('/', methods=['GET'])
def index():
	# Main Page
	return render_template('index.html')



@app.route('/predict', methods=['GET','POST'])
def upload():
	if request.method == 'POST':
		# Get the file from post request
		f = request.files['file']

		# Save the file to ./uploads
		basepath = os.path.dirname(__file__)
		file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
		f.save(file_path)

		# Make Prediction
		pred = model_predict(file_path, model, net)

		return pred

	return None


if __name__ == '__main__':
	app.run(debug=True)













