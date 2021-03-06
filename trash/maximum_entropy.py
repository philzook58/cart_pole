import cv2
import numpy as np

cap = cv2.VideoCapture(0)


# mouse callback function



def draw_circle(event,x,y,flags,param):
    global frame
    if event == cv2.EVENT_LBUTTONDOWN:
        #cv2.circle(frame,(x*2,2*y),20,(255,0,0),10)

        cv2.rectangle(frame, (x-10,y - 10 ), (x + 10, y+10),(255,0,0),10 )
        print("click")
        print(x,y)
        x= int(x)
        y = int(y)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        subframe = hsv[ x-10 : x+10, y-10 : y + 10]

        outbox = np.zeros(255)
        inbox = np.zeros(255)

        unique, counts = np.unique(hsv[:,:,0], return_counts=True)
        outbox[unique]=counts
        print(unique)
        unique, counts = np.unique(subframe[:,:,0], return_counts=True)

        inbox[unique] = counts
        outbox = outbox - inbox
        #print(outbox)

        bestupper = 0
        bestlower = 0
        bestscore = 10000000000
        totinbox = np.sum(inbox)
        totoutbox = np.sum(outbox)
        for upper in range(255):
            for lower in range(upper):
                posinbox = np.sum(inbox[lower:upper])

                neginbox = totinbox - posinbox 
                negoutbox = np.sum(outbox[lower:upper])
                posoutbox = totoutbox - negoutbox 
                def entr(x):
                    return - x * np.log(x+0.000000001)
                score = entr(negoutbox/totoutbox) + entr(posoutbox/totoutbox) + entr(neginbox/totinbox) + entr(posinbox/totinbox)
                #score = posinbox/totinbox - negoutbox/totinbox
                if score < bestscore:
                    bestupper = upper
                    bestscore = score
                    bestlower = lower
        print(bestlower, bestupper, bestscore)


        #print frame[x,y]

# Create a black image, a window and bind the function to window
#img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('mask')
cv2.setMouseCallback('mask',draw_circle)

# Take each frame
_, frame = cap.read()

#frame = cv2.resize(frame,None,fx=.25, fy=.25, interpolation = cv2.INTER_CUBIC)

print(frame.shape)

x = np.arange(frame.shape[1])
y = np.arange(frame.shape[0])
xv, yv = np.meshgrid(x, y, sparse=False)


def getAveragePosColor(frame, color):
    lower = np.array([color - 15,50,50])
    upper = np.array([color + 15,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower, upper)

    N = np.sum(mask) + 1
    xavg = np.sum( mask * xv) / N
    yavg = np.sum( mask * yv) / N
    return N, x ,y



while(1):

    # Take each frame
    _, frame = cap.read()

    #frame = cv2.resize(frame,None,fx=.25, fy=.25, interpolation = cv2.INTER_CUBIC)
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #hsv = cv2.resize(hsv,None,fx=.25, fy=.25, interpolation = cv2.INTER_CUBIC)
    #small = cv2.pyrDown(hsv)
    # define range of blue color in HSV

    #342 for magneta * .7 = 240 -- nope 180 works well
    #42 for gold * .7 = 30
    #11 for orange * .7 = 8
    # 135for green * .7 = 95 # 50 70 works before # 60 is perfect
    color = 180
    #print(color)
    lower_blue = np.array([color-15,100,50])
    upper_blue = np.array([color+15,200,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    #res = cv2.bitwise_and(frame,frame, mask= mask)
    N = np.sum(mask)
    if N >= 30 :
        xavg = np.sum( mask * xv) / N
        yavg = np.sum( mask * yv) / N
        #print("x:" + str(xavg) + " y: " + str(yavg))
    #cv2.imshow('frame',frame)
    #cv2.imshow('mask',cv2.pyrDown(frame))
    #cv2.imshow('res',res)
    #cv2.imshow('res',mask)
    k = cv2.waitKey(100) & 0xFF
    if k == 27:
        break
    cv2.imshow('mask',frame)

cv2.destroyAllWindows()