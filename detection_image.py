# import the necessary packages
import numpy as np
import argparse
import cv2
from shapely.geometry import Polygon

 


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])



# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

#draw polygon
# top-left, buttom-left, buttom-right, top-right
pts = np.array([[90,250],[90,350],[250,350],[250,250]], np.int32)
pts_reshape = pts.reshape((-1,1,2))
image = cv2.polylines(image,[pts_reshape],True,(0,255,255))
# cv2.imshow("pic",image)
# cv2.waitKey(0)
# print(pts)

# pass the blob through the net
# work and obtain the detections and
# predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()
# loop over the detections
for i in np.arange(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the prediction
	confidence = detections[0, 0, i, 2]
	# filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
	if confidence > args["confidence"]:
		# extract the index of the class label from the `detections`,
		# then compute the (x, y)-coordinates of the bounding box for
		# the object
		idx = int(detections[0, 0, i, 1])
		print(idx)
		if ( idx==7 or idx==6 or idx==14 ):


			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			#calculate polygon coordinatef from detection points
			ax=startX
			ay=startY
			bx=startX
			by=endY
			cx=endX
			cy=endY
			dx=endX
			dy=startY
			
			detected_pts = np.array([[ax,ay],[bx,by],[cx,cy],[dx,dy]], np.int32)
			detected_pts_reshape = detected_pts.reshape((-1,1,2))
			image = cv2.polylines(image,[detected_pts_reshape],True,(0,0,255))

			# find collision ````````````````````````````````````````````

			p1=Polygon(pts)
			p2=Polygon(detected_pts)
			p3=p2.intersection(p1)
			print(p3) # result: POLYGON ((0.5 0.5, 1 1, 1 0, 0.5 0.5))
			print(p3.area*100/p1.area,"%") # result: 0.25


			# ``````````````````````````````````````````````````````````

			
			# display the prediction
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			print("[INFO] {}".format(label))
			cv2.rectangle(image, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(image, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)

# USAGE
# python deep.py --image images/example_01.jpg --prototxt MobileNetSSD.prototxt.txt --model NetSSD.caffemodel


