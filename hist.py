import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
plt.figure()
plt.title('Histogram')
plt.xlabel('Bins')
plt.ylabel('Pixels')

def eq_hist(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    channels[0] = cv2.equalizeHist(channels[0])
    ycrcb = cv2.merge(channels)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)

while True:
    # Read the frame
    _, img = cap.read()

    # Equalize Hist
    img = eq_hist(img)

    colors = ('b','g','r')
    channels = cv2.split(img)
    plt.clf()
    
    for (channel, color) in zip(channels, colors):
        hist= cv2.calcHist([channel], [0], None, [256], [0,256])
        plt.plot(hist, color = color)
        plt.xlim([0, 256])

    plt.draw()
    plt.pause(0.001)
    cv2.imshow('study', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()


