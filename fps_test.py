import cv2
import numpy as np
import time
cap = cv2.VideoCapture( 3)


#cap.

cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 240)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 320)
cap.set(cv2.cv.CV_CAP_PROP_FPS, 125)

while True:
	   # Start time
    start = time.time()
     


    ret, frame = cap.read()
 
     
    # End time
    end = time.time()
    cv2.imshow('frame',frame)
    print(1/(end-start))

    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
	   break