# Import libraries
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from flask import Flask, render_template, Response
from imutils.video import VideoStream
from imutils.video import WebcamVideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os




# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str, default="face_detector", help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str, default="mask_detector.model", help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Load the face detector model from the disk
print("Face detector model loading...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],"res10_300x300_ssd_iter_140000.caffemodel"])

faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)


# Load the face mask detector model that we have trained
print("Face Mask detector model loading...")
maskNet = load_model(args["model"])



app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')


def detect_and_predict_mask(frame, faceNet, maskNet):
	# Grab the dimensions of the frame and then construct a blob from it
	(h,w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104.0, 177.0, 123.0))

	# Pass the blob through network and get the predictions
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# Initialise the list of faces, their locations and list of predictions from our face mask network
	faces = []
	locs = []
	preds = []


	# Loop over the detections
	for i in range(0, detections.shape[2]):
		# Find the confidence
		confidence = detections[0,0,i,2]

		# Filter out weak detections by ensuring the confidence is greater than the minimum confidence
		if confidence > args["confidence"]:
			# Compute the coordinates of the bounding box
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Ensure the bounding boxes fall within the dimensions of the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))


			# Extract the ROI, convert it from BGR to RGB, ordering and resizing it to 224X224
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# Add the face and bounding boxes to their respective lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))



	# Make prediction if one or more face is detected
	if len(faces) > 0:
		# Making batch prediction on all faces not one by one 
		faces = np.array(faces, dtype='float32')
		preds = maskNet.predict(faces, batch_size=32)

	return (locs, preds)



def gen():
	# Initialize the video stream
	print("Starting the video stream...")
	vs = WebcamVideoStream(src=0).start()
	time.sleep(1.0)

	# vs = VideoStream(src=0).start()
	# time.sleep(2.0)
	while True:	
		frame = vs.read()
		#print(frame)
		frame = imutils.resize(frame, width=800)

		#rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		# Detect the faces and masks in the video frames
		(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

		# Loop over the detected face locations and their probability of mask wearing
		for (box, pred) in zip(locs, preds):
			(startX, startY, endX, endY) = box
			(mask, withoutMask) = pred

			# determine the class label and color we'll use to draw the bounding box and text
			label = "Mask" if mask > withoutMask else "No Mask"
			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
			# include the probability in the label
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
			# display the label and bounding box rectangle on the output frame
			cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 4)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

			imgencode=cv2.imencode('.jpg',frame)[1]
			stringData = imgencode.tostring()
			yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                stringData + b'\r\n')
	     
			

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                        mimetype = "multipart/x-mixed-replace; boundary=frame")


if __name__=="__main__":
    #app.run(debug=True)
	app.run(host='0.0.0.0', port='5000', debug=True)





