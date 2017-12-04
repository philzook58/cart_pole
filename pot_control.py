import cv2
import numpy as np
import pickle
from sabretooth_command import CartCommand
from time import time, sleep
import Queue
from pyfirmata import Arduino, util


cart = CartCommand(port="/dev/ttyACM0")

setpoint = 0.74




board = Arduino('/dev/ttyUSB0')
it = util.Iterator(board)
it.start()
analog_0 = board.get_pin('a:0:i')
sleep(0.5)
old_theta = analog_0.read() - setpoint


itheta = 0

start = 0

def nothing(x):
	pass


cv2.namedWindow('PID', cv2.WINDOW_NORMAL)
cv2.resizeWindow('PID', 600,200)


cv2.createTrackbar('P','PID',0,200000,nothing)
cv2.setTrackbarPos('P', 'PID', 100000)

cv2.createTrackbar('I','PID',0,20000,nothing)
cv2.setTrackbarPos('I', 'PID', 10000)

cv2.createTrackbar('D','PID',0,200000,nothing)
cv2.setTrackbarPos('D', 'PID', 100000)



cv2.createTrackbar('SetPoint','PID',0, 100,nothing)
cv2.setTrackbarPos('SetPoint', 'PID', 50)

cv2.createTrackbar('powerdecay','PID',0,1000,nothing)
cv2.setTrackbarPos('powerdecay', 'PID', 0)
'''
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
'''

#def analogToAngle(pinval):
#	zeropoint = 0.741

#	pinval *
old_command = 0

dtheta = np.pi
while True:

	start = time()
	theta = analog_0.read()
	dsetpoint = (cv2.getTrackbarPos('SetPoint','PID') - 50)/1000.0
	theta = theta - setpoint - dsetpoint
	dtheta = theta - old_theta
	old_theta = theta
	
	diff = theta
	#diff = np.random.uniform(-.05,.05)
	print(diff)
	if cart.enabled:
		itheta += diff
		#itheta = max(-np.pi/4, itheta)
		#itheta = min(np.pi/4, itheta)

	kp = 300
	kd = 0
	ki = 0

	kp = cv2.getTrackbarPos('P','PID') - 100000
	ki = cv2.getTrackbarPos('I','PID') - 10000
	kd = cv2.getTrackbarPos('D','PID') - 100000
	decay = (cv2.getTrackbarPos('powerdecay','PID') - 0)/100

	command = (kp * diff) + (kd * dtheta) + (ki * itheta)
	command += -decay * old_command
	
	command = min(max(command,-2046), 2046)
	old_command = command
	print("p=%d\ti=%d\td=%d"%(kp*diff, ki*itheta, kd*dtheta))
	print command



	cart.setSpeed(command)



	key = cv2.waitKey(1)
	if key & 0xFF == 32:
		print("toggle enable")
		itheta = 0
		cart.toggleEnable()
	elif key & 0xFF == ord('q'):
		it.stop()
		#analyzer.save()
		break
#	elif key & 0xFF == ord('r'):
#		print("reset")
#		reset()

	#print time() - start

