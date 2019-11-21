import cv2
import imutils

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()
    img = imutils.resize(img, width=min(640, img.shape[1]))
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray  = cv2.equalizeHist(gray)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Draw the rectangle around each face
    for face in faces:
        x, y, w, h = face
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        upper_gray = gray[y:(y+int(h*2/3)), x:x+w]
        upper_color = img[y:(y+int(h*2/3)), x:x+w]
        eyes = eye_cascade.detectMultiScale(upper_gray, 1.05, 3)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(upper_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        lower_gray = gray[y+int(h*2/3):y+h, x:x+w]
        lower_color = img[y+int(h*2/3):y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(lower_gray, 1.2, 3)
        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(lower_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)

    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()
