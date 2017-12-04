import cv2
import numpy as np
import pickle
from sabretooth_command import CartCommand
from image_analyzer_pseye import ImageAnalyzer
from time import time
import Queue

old_data = []
data = []

analyzer = ImageAnalyzer()

cart = CartCommand()

commandqueue = Queue.Queue()

for i in range(5):
	commandqueue.put(0)

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

def getData():	
	x, dx, theta, dtheta = analyzer.analyzeFrame()
	xpole = np.cos(theta)
	ypole = np.sin(theta)
	return x, xpole, ypole

command = 0
commandStep = 100

old_x, old_xpole, old_ypole = getData()

while True:

	action = (2*np.random.randint(2) - 1)
	command += commandStep * action
	command = min(max(command,-2046), 2046)

	print(len(old_data))

	start = time()

	x, xpole, ypole = getData()
	state = [x, x-old_x, xpole, xpole-old_xpole, ypole, ypole-old_ypole] + list(commandqueue.queue)
	if cart.enabled:
		data.append({
			"state": state,
			"action": action
			})
		try:
			data[-2]["next_state"] = state
			print(data[-2])
		except IndexError:
			pass


	if x < 0.2:
		command = min(command,0)
	if x > 0.8:
		command = max(command, 0)

	cart.setSpeed(command)
	commandqueue.put(command)
	commandqueue.get()
	print(commandqueue.queue)




	key = cv2.waitKey(1)
	if key & 0xFF == 32:
		print("toggle enable")
		old_data += data[0:-2]
		data = []
		itheta = 0
		cart.toggleEnable()
	elif key & 0xFF == ord('q'):
		analyzer.save()
		break
	elif key & 0xFF == ord('r'):
		print("reset")
		reset()

old_data += data[0:-2]
assert(all([len(d.keys()) == 3 for d in old_data]))
pickle.dump(old_data[0:-2], open("data%d.p"%int(time()),"w"))