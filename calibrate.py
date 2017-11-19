import cv2
import numpy as np
import pickle

left = right = None


def drawCircle(frame,x,y):
    cv2.circle(frame,(x,y),20,(255,0,0),2)

def calibLeft(event, x, y, flags, param):
    global left
    if event == cv2.EVENT_LBUTTONDOWN:
        left = (x, y)
        drawCircle(frame, x, y)
        cv2.destroyAllWindows()


def calibRight(event, x, y, flags, param):
    global right
    if event == cv2.EVENT_LBUTTONDOWN:
        right= (x, y)
        drawCircle(frame, x, y)
        cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
cv2.imshow("calibrate", frame)
cv2.setMouseCallback("calibrate", calibLeft)
print("click left end")
cv2.waitKey(0) & 0xFF

cv2.imshow("calibrate", frame)
cv2.setMouseCallback("calibrate", calibRight)
print("click right end")
cv2.waitKey(0) & 0xFF

cv2.imshow("calibrate", frame)
cv2.waitKey(0) & 0xFF

print(left,right)

pickle.dump({"left":left, "right":right}, open("calib.p", "w"))