
import mtcnn
import cv2
import numpy as np
import imutils
import json

cap = cv2.VideoCapture(0)

detector = mtcnn.mtcnn.MTCNN()

isTracking = False
cntFrame = 0
#tracker = cv2.TrackerKCF_create()
#tracker = cv2.TrackerCSRT_create()
tracker = cv2.TrackerMOSSE_create()

while True:
    # Read the frame
    _, img = cap.read()

    img = imutils.resize(img, width=min(640, img.shape[1]))

    # use tracker to improve performance
    if isTracking and cntFrame < 30:
        cntFrame += 1
        isTracking, box = tracker.update(img)
        x = int(box[0])
        y = int(box[1])
        w = int(box[2])
        h = int(box[3])
        if isTracking:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    else:
        isTracking = False
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(img_rgb)
        for face in faces:
            [x,y,w,h] = face['box']
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            #cv2.circle(img, face['keypoints']['nose'], 5, (255,255,0))
            #cv2.circle(img, face['keypoints']['right_eye'], 5, (0,255,255))
            #cv2.circle(img, face['keypoints']['left_eye'], 5, (0,255,255))
            #v2.circle(img, face['keypoints']['mouth_right'], 5, (255,0,255))
            #cv2.circle(img, face['keypoints']['mouth_left'], 5, (255,0,255))
            #cv2.putText(img, str(face['confidence']), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
            if isTracking == False:
                tracker = cv2.TrackerKCF_create()
                tracker.init(img, (x,y,w,h))
                isTracking = True
                cntFrame = 0

    # Display
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()
