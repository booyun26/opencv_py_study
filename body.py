
from imutils.object_detection import non_max_suppression

import cv2
import imutils
import numpy as np
import random

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(0)

colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255)]

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes	
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

	# compute the width and height of the bounding box
	w = np.maximum(0, xx2 - xx1 + 1)
	h = np.maximum(0, yy2 - yy1 + 1)

	# compute the ratio of overlap
	overlap = (w * h) / area[idxs[:last]]

	# delete all indexes from the index list that have
	idxs = np.delete(idxs, np.concatenate(([last],
	np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

while True:
    # Read the frame
    _, img = cap.read()

    img = imutils.resize(img, width=min(640, img.shape[1]))
    orig = img.copy()

    (rects, weights) = hog.detectMultiScale(img, winStride=(8, 8),
		padding=(8, 8), scale=1.05)

    i = 0
    for (x,y,w,h) in rects:
        cv2.rectangle(orig, (x, y), (x+w, y+h), colors[i%6], 2)
        i+=1

    # apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
    #nrects = np.array([[x,y,w+x,h+y] for (x,y,w,h) in rects])
    #pick = non_max_suppression_fast(nrects, 0.65)
	# draw the final bounding boxes
    #for (x,y,x1,y1) in pick:
    #    cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 2)
    # Display
    #hor_merge = np.hstack((orig,img))
    #cv2.imshow('people detection',hor_merge)
    cv2.imshow('people detection',orig)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()

