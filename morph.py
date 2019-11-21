import cv2
import imutils
import numpy as np

def nothing(x):
    pass

def getKernel(ktype, ksize):
    ktypes = {
        0 : cv2.MORPH_RECT,
        1 : cv2.MORPH_ELLIPSE,
        2 : cv2.MORPH_CROSS
    }

    return cv2.getStructuringElement(ktypes.get(ktype), (ksize, ksize))

def getMorphType(pos):
    mtypes = {
        0 : cv2.MORPH_GRADIENT,
        1 : cv2.MORPH_OPEN,
        2 : cv2.MORPH_CLOSE,
        3 : cv2.MORPH_DILATE,
        4 : cv2.MORPH_ERODE,
        5 : cv2.MORPH_BLACKHAT,
        6 : cv2.MORPH_TOPHAT,
        7 : cv2.MORPH_HITMISS
    }
    mtypes_str = {
        0 : 'GRADIENT',
        1 : 'OPEN',
        2 : 'CLOSE',
        3 : 'DILATE',
        4 : 'ERODE',
        5 : 'BLACKHAT',
        6 : 'TOPHAT',
        7 : 'HITMISS'
    }
    return (mtypes.get(pos), mtypes_str.get(pos))

cap = cv2.VideoCapture(0)
cv2.namedWindow('study')

cv2.createTrackbar('morph type', 'study', 0, 7, nothing)
cv2.createTrackbar('kernel size', 'study', 3, 9, nothing)
cv2.setTrackbarMin('kernel size', 'study', 2)
cv2.createTrackbar('kernel type', 'study', 0, 2, nothing)

while True:
    # Read the frame
    _, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray  = cv2.equalizeHist(gray)

    # morphology
    mtype_pos = cv2.getTrackbarPos('morph type', 'study')
    ktype_pos = cv2.getTrackbarPos('kernel type', 'study')
    ksize_pos = cv2.getTrackbarPos('kernel size', 'study')

    (m_type, m_str) = getMorphType(mtype_pos)
    kernel = getKernel(ktype_pos, ksize_pos)

    morph = cv2.morphologyEx(gray, m_type, kernel)

    # blurring and thresholding
    #blur = cv2.GaussianBlur(morph, (5,5), 0)
    #_, morph = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)

    # Display
    #hstack = np.hstack((img, morph))
    hstack = np.hstack((img, cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)))
    cv2.putText(hstack, m_str, (10+hstack.shape[1]//2,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
    cv2.imshow('study', hstack)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()
