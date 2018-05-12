import cv2
import numpy as np
import pickle
from cartcommand import CartCommand

top = bot = (0, 0)
left = right = None
pole = cart = None

calib = pickle.load(open("calib.p","r"))
settings = {'h_top':0, 'h_bot':0}
try:
	settings = pickle.load(open("settings.p","r"))
except IOError:
	print("Nonsettings file found")
	pass\

left = calib["left"]
right = calib["right"]

def nothing(x):
	pass

cv2.namedWindow('frame')
cv2.namedWindow('mask top')
cv2.namedWindow('mask bot')
cv2.namedWindow('controls', cv2.WINDOW_NORMAL)
cv2.resizeWindow('controls', 600,200)

colorpanel = np.ones((10,180,3), dtype=np.uint8) * 255
colorpanel[:,:,0] = np.arange(180).reshape((1,-1))
colorpanel = np.repeat(colorpanel, 3, axis=1)
colorpanel = cv2.cvtColor(colorpanel, cv2.COLOR_HSV2BGR)
cv2.imshow('controls',colorpanel)

cv2.createTrackbar('top','controls',0,180,nothing)
cv2.setTrackbarPos('top', 'controls', settings["h_top"])

cv2.createTrackbar('bot','controls',0,180,nothing)
cv2.setTrackbarPos('bot', 'controls', settings["h_bot"])

cap = cv2.VideoCapture(1)


def getBGRFromH(H):
	color = cv2.cvtColor(np.uint8([[[H,255,255]]]), cv2.COLOR_HSV2BGR)[0][0]
	return (int(color[0]),int(color[1]),int(color[2]))


def getMask(frame, color):
	lower = np.array([color,50,50])
	upper = np.array([color + 16 - 1,255,255])

	# Threshold the HSV image to get only blue colors
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, lower, upper)
	erodenum = 2
	mask = cv2.erode(mask, None, iterations=erodenum)
	mask = cv2.dilate(mask, None, iterations=erodenum)

	return mask

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
		return None, None #(0,0) #defaults


old_x = 0
old_theta = 0
itheta = 0

cart = CartCommand()

while True:

	ret, frame = cap.read()

	h_top = cv2.getTrackbarPos('top','controls')
	h_bot = cv2.getTrackbarPos('bot','controls')

	mask_top = getMask(frame,h_top)
	xavg_top, yavg_top = getBiggestContour(mask_top)
	if not xavg_top is None and not yavg_top is None:
		top = (int(xavg_top), int(yavg_top))

	mask_bot = getMask(frame,h_bot)
	xavg_bot, yavg_bot = getBiggestContour(mask_bot)
	if not xavg_bot is None and not yavg_bot is None:
		bot = (int(xavg_bot), int(yavg_bot))

	cv2.circle(frame,top,20,getBGRFromH(h_top),10)
	cv2.circle(frame,bot,20,getBGRFromH(h_bot),10)
	cv2.line(frame,top, bot,(255,255,255),5)

	cv2.circle(frame,left,5,(255,255,255),5)
	cv2.circle(frame,right,5,(255,255,255),5)
	cv2.line(frame,left, right,(255,255,255),5)
	cv2.imshow('frame',cv2.pyrDown(frame))
	cv2.imshow('mask top', cv2.pyrDown(mask_top))
	cv2.imshow('mask bot', cv2.pyrDown(mask_bot))

	# re-center coordinates at left base point
	l = np.array(left)
	r = np.array(right)-l
	t = np.array(top)-l
	b = np.array(bot)-l
	l = l - l

	# reflect coordinates for a bottom-left system
	r = np.dot(r, np.array([[1,0],[0,-1]]))
	t = np.dot(t, np.array([[1,0],[0,-1]]))
	b = np.dot(b, np.array([[1,0],[0,-1]]))

	# find the baseline angle and make a rotation matrix
	a = np.arctan(float(r[1])/float(r[0]))
	rot = np.array([[np.cos(a), -np.sin(a)],[np.sin(a), np.cos(a)]])

	# rotate the coordinates flat
	r = np.dot(r, rot)
	t = np.dot(t, rot)
	b = np.dot(b, rot)

	l = r[0]
	r = r/l
	t = t/l
	b = b/l

	#print(l,r,t,b)

	# find cart position and velocity
	x = b[0]
	dx = x-old_x
	old_x = x

	# find pole vector, angle, and angular velocity
	pole = t-b
	theta = np.arctan2(pole[1],pole[0])
	dtheta = theta-old_theta
	old_theta = theta

	#print(x,theta)


	diff = theta - np.pi/2

	if cart.enabled:
		itheta += diff
		itheta = max(-np.pi/4, itheta)
		itheta = min(np.pi/4, itheta)

	kp = 300
	kd = 300
	ki = 0
	if 0.2 < x < 0.8: 
		print(-(kp * diff) - (kd * dtheta) - (ki * itheta))
		cart.move(-(kp * diff) - (kd * dtheta) - (ki * itheta), 300)

	key = cv2.waitKey(1)
	if key & 0xFF == 32:
		print("toggle enable")
		itheta = 0
		cart.toggleEnable()
	elif key & 0xFF == ord('q'):
		pickle.dump({"h_top":h_top,"h_bot":h_bot}, open("settings.p","w"))
		break

