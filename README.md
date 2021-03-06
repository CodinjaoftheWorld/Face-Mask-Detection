# Face-Mask-Detection-from-Images-and-Video-feed

This repository is showing how to detect the masks on the live video stream using openCV.

The original model to detect mask on faces is trained using mobilenetV2 architecture. I am thankful to Adrian Rosebrock for his excellent blogs and tutorials on computer vision. I picked one of his tutorial on the face mask detection on images and video feed and just created an API using Flask.


I highly recommed to work on the below link for model training and mask detections for Images and Video Feed.
Link to the tutorial: https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/



Part 1: Detect the Face Mask from Video Feed

Setps:
  1. Clone the repository, go to directory ‘Detect from video stream’ and run app.py file.

  2. Go to http://0.0.0.0:5000/. Grab your mask to check the detection.



Part 2: Predict the Face Mask on Image

Setps:
1. Clone the repository, go to directory ‘Detect from Images’ and run app.py file.

2. Go to http://0.0.0.0:5000/ or http://127.0.0.1:5000/

3. Click on Choose button to select the images.

4. Click on Predict button to see ‘Mask’ or ‘No Mask’ remark. Mask remark denotes that the images has human with facemask and No Mask remarl denotes that their is no mask in the image. 

![](Detect%20from%20Images/face_Mask_Detector_Web_GUI.png)


Alert: Follow the link and get tesoflow installed in your CPU ot GPU. Please create the virtual environment and always run the codes in venv.
https://www.pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-ubuntu/

