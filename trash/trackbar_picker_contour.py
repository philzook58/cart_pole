import cv2
import numpy as np

cap = cv2.VideoCapture(0)

cv2.namedWindow('frame')
cv2.namedWindow('controls', cv2.WINDOW_NORMAL)
cv2.resizeWindow('controls', 600,200)

colorpanel = np.ones((10,255,3), dtype=np.uint8) * 255
print(colorpanel)
colorpanel[:,:,0] = np.arange(255).reshape((1,-1))
colorpanel = np.repeat(colorpanel, 3, axis=1)
colorpanel = cv2.cvtColor(colorpanel, cv2.COLOR_HSV2BGR)
cv2.imshow('controls',colorpanel)

def nothing(x):
    print(x)
    pass

cv2.createTrackbar('R','controls',0,255,nothing)
cv2.namedWindow('mask')
_, frame = cap.read()
print(frame.shape)


x = np.arange(frame.shape[1])
y = np.arange(frame.shape[0])
xv, yv = np.meshgrid(x, y, sparse=False)







def getAveragePosColor(frame, color):
    lower = np.array([color,50,50])
    upper = np.array([color + 16 - 1,255,255])

    # Threshold the HSV image to get only blue colors
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    erodenum = 2
    mask = cv2.erode(mask, None, iterations=erodenum)
    mask = cv2.dilate(mask, None, iterations=erodenum)

    N = np.sum(mask) + 1
    xavg = np.sum( mask * xv) / N
    yavg = np.sum( mask * yv) / N
    return N, xavg , yavg, mask

def getBiggestContour(mask):
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
 
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return center
    else:
        return (0,0) #defaults
        # only proceed if the radius meets a minimum size
        #if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            #cv2.circle(frame, (int(x), int(y)), int(radius),
            #    (0, 255, 255), 2)
            #cv2.circle(frame, center, 5, (0, 0, 255), -1)


while(1):

    # Take each frame
    k = cv2.waitKey(100) & 0xFF
    if k == 27:
        break
    _, frame = cap.read()
    r = cv2.getTrackbarPos('R','controls')
    N, xavg, yavg, mask = getAveragePosColor(frame,r)
    xavg, yavg = getBiggestContour(mask)
    cv2.circle(frame,(int(xavg),int(yavg)),20,(255,0,0),10)
    cv2.imshow('frame',cv2.pyrDown(frame))
    cv2.imshow('mask', cv2.pyrDown(mask))


cv2.destroyAllWindows()