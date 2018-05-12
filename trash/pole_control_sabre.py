import cv2
import numpy as np
import pickle
from sabretooth_command import CartCommand
from image_analyzer_pseye import ImageAnalyzer
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


cv2.createTrackbar('P','PID',0,200000,nothing)
cv2.setTrackbarPos('P', 'PID', 100000)

cv2.createTrackbar('I','PID',0,16000,nothing)
cv2.setTrackbarPos('I', 'PID', 8000)

cv2.createTrackbar('D','PID',0,200000,nothing)
cv2.setTrackbarPos('D', 'PID', 100000)


cv2.createTrackbar('SetPoint','PID',0, 100,nothing)
cv2.setTrackbarPos('SetPoint', 'PID', 50)

cv2.createTrackbar('powerdecay','PID',0,1000,nothing)
cv2.setTrackbarPos('powerdecay', 'PID', 0)

def reset():	
	x = 0
	cart.enabled = True
	while not 0.4 < x < 0.6:
		x, dx, theta, dtheta = analyzer.analyzeFrame()

		command = 1000 * np.sign(x-0.5)
		command = min(max(command,-2046), 2046)
		print(command)

		cart.setSpeed(command)
		cv2.waitKey(1)

	cart.setSpeed(0)
	cart.enabled = False




while True:

	start = time()

	x, dx, theta, dtheta = analyzer.analyzeFrame()
	
	dsetpoint = (cv2.getTrackbarPos('SetPoint','PID') - 50)/1000.0

	diff = theta - np.pi/2 - dsetpoint #0.004

	#diff = np.random.uniform(-.05,.05)
	print(diff)
	if cart.enabled:
		itheta += diff
		itheta = max(-np.pi/4, itheta)
		itheta = min(np.pi/4, itheta)

	kp = 300
	kd = 0
	ki = 0

	kp = cv2.getTrackbarPos('P','PID') - 100000
	ki = cv2.getTrackbarPos('I','PID') - 8000
	kd = cv2.getTrackbarPos('D','PID') - 100000
	decay = (cv2.getTrackbarPos('powerdecay','PID') - 0)/100

	command = (kp * diff) + (kd * dtheta) + (ki * itheta)
	command = min(max(command,-2046), 2046)
	print("p=%d\ti=%d\td=%d"%(kp*diff, ki*itheta, kd*dtheta))
	print command


	if 0.2 < x < 0.8: 
		#print(-(kp * diff) - (kd * dtheta) - (ki * itheta))
		cart.setSpeed(command)
	else:
		cart.setSpeed(0)


	key = cv2.waitKey(1)
	if key & 0xFF == 32:
		print("toggle enable")
		itheta = 0
		cart.toggleEnable()
	elif key & 0xFF == ord('q'):
		analyzer.save()
		break
	elif key & 0xFF == ord('r'):
		print("reset")
		reset()

	#print time() - start

