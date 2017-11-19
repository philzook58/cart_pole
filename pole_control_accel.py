import cv2
import numpy as np
import pickle
from cartcommand import CartCommand
from image_analyzer import ImageAnalyzer
from time import time
import Queue

analyzer = ImageAnalyzer()

cart = CartCommand()

itheta = 0

start = 0

def nothing(x):
	pass


cv2.namedWindow('PID', cv2.WINDOW_NORMAL)
cv2.resizeWindow('PID', 600,200)


cv2.createTrackbar('P','PID',0,1000,nothing)
cv2.setTrackbarPos('P', 'PID', 0)

#cv2.createTrackbar('I','PID',0,1000,nothing)
#cv2.setTrackbarPos('I', 'PID', 0)

cv2.createTrackbar('D','PID',0,2000,nothing)
cv2.setTrackbarPos('D', 'PID', 1000)



while True:

	start = time()

	x, dx, theta, dtheta = analyzer.analyzeFrame()

	diff = theta - np.pi/2

	#diff = np.random.uniform(-.05,.05)

	if cart.enabled:
		itheta += diff
		itheta = max(-np.pi/4, itheta)
		itheta = min(np.pi/4, itheta)

	kp = 300
	kd = 0
	ki = 0

	kp = cv2.getTrackbarPos('P','PID')
	#ki = cv2.getTrackbarPos('I','PID')
	kd = cv2.getTrackbarPos('D','PID') - 1000

	command = -(kp * diff) - (kd * dtheta) - (ki * itheta)
	print command

	if 0.2 < x < 0.8: 
		#print(-(kp * diff) - (kd * dtheta) - (ki * itheta))
		cart.move(command, 300)


	key = cv2.waitKey(1)
	if key & 0xFF == 32:
		print("toggle enable")
		itheta = 0
		cart.toggleEnable()
	elif key & 0xFF == ord('q'):
		analyzer.save()
		break

	#print time() - start

