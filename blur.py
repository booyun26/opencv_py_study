import cv2
import imutils
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
cv2.namedWindow('blur')

cv2.createTrackbar('blur type', 'blur', 0, 3, nothing)
cv2.createTrackbar('blur level', 'blur', 1, 3, nothing)
cv2.setTrackbarMin('blur level', 'blur', 1)

while True:
    # Read the frame
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = gray
    title = 'Average '

    type_pos = cv2.getTrackbarPos('blur type', 'blur')
    level_pos = cv2.getTrackbarPos('blur level', 'blur')
    if type_pos == 0:
        title = 'Bilateral '
        if level_pos == 1:
            blur = cv2.bilateralFilter(blur, 5, 21, 21)
        elif level_pos == 2:
            blur = cv2.bilateralFilter(blur, 7, 51, 51)
        elif level_pos == 3:
            blur = cv2.bilateralFilter(blur, 9, 101, 101)
    elif type_pos == 1:
        title = 'Gaussian '
        if level_pos == 1:
            blur = cv2.GaussianBlur(blur, (3,3), 0)
        elif level_pos == 2:
            blur = cv2.GaussianBlur(blur, (5,5), 0)
        elif level_pos == 3:
            blur = cv2.GaussianBlur(blur, (7,7), 0)
    elif type_pos == 2:
        title = 'Median '
        if level_pos == 1:
            blur = cv2.medianBlur(blur, 3)
        elif level_pos == 2:
            blur = cv2.medianBlur(blur, 5)
        elif level_pos == 3:
            blur = cv2.medianBlur(blur, 7)
    else:
        if level_pos == 1:
            blur = cv2.blur(blur, (3,3))
        elif level_pos == 2:
            blur = cv2.blur(blur, (5,5))
        elif level_pos == 3:
            blur = cv2.blur(blur, (7,7))

    title += str(level_pos)

    _, blur = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
    # Display
    hstack = None
    if blur.ndim == 2:
        hstack = np.hstack((img, cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)))
    else:
        hstack = np.hstack((img, blur))
    cv2.putText(hstack, title, (10+hstack.shape[1]//2,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
    cv2.imshow('blur', hstack)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()

