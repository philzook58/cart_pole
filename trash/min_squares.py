import cv2
import numpy as np

cap = cv2.VideoCapture(0)

imagescale = 1

def mouseHandler(event,x,y,flags,param):
    global frame, globalcolor
    x = imagescale * x
    y = imagescale * y
    if event == cv2.EVENT_LBUTTONDOWN:   
        # The idea is to try a range of color cutoffs and see which one does best by the score
        #the score is a least squares of masked pixels from your click positions
        # weighted by to total number to try and balance getting a lot of pixels, but not wrong ones.
        #Is this really any better than just taking the color you click on? 
        print("click")
        print(x,y)
        bestcolor = 0
        bestscore = 1000000000000000000000

        for colordiv16 in range(16):
                color = colordiv16 * 16
                N, xavg, yavg, mask = getAveragePosColor(frame, color)
                if N >= 100:
                    # the power of 1.5 just seems to work better than 2 or 1. I dunno.
                    score = np.sum(((x - xv)**2 + (y - yv)**2) * mask)/N**1.5 #+ (N - 100*100)**2
                    if score < bestscore:
                        bestcolor = color
                        bestscore = score
        #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #bestcolor = hsv[int(y),int(x),0]
        print(bestcolor, bestscore)
        globalcolor = bestcolor

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', mouseHandler)
cv2.namedWindow('mask')
_, frame = cap.read()
print(frame.shape)


x = np.arange(frame.shape[1])
y = np.arange(frame.shape[0])
xv, yv = np.meshgrid(x, y, sparse=False)


globalcolor = 0





def getAveragePosColor(frame, color):
    lower = np.array([color,50,50])
    upper = np.array([color + 16 - 1,255,255])

    # Threshold the HSV image to get only blue colors
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    N = np.sum(mask) + 1
    xavg = np.sum( mask * xv) / N
    yavg = np.sum( mask * yv) / N
    return N, x , y, mask



while(1):

    # Take each frame
    _, frame = cap.read()

    N, xavg, yavg, mask = getAveragePosColor(frame,globalcolor)
    cv2.imshow('frame',cv2.pyrDown(frame))
    cv2.imshow('mask', cv2.pyrDown(mask))

    k = cv2.waitKey(100) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()