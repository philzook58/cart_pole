from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers, regularizers

import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt

import cv2
from sabretooth_command import CartCommand
from image_analyzer_pseye import ImageAnalyzer
from time import time, sleep
import Queue
from keras.models import load_model


def makeModel():
	reg = 1e-13
	model = Sequential([
	Dense(12, input_shape=(6,), kernel_regularizer=regularizers.l2(reg)),
	Activation('relu'),
	Dense(12, kernel_regularizer=regularizers.l2(reg)),
	Activation('relu'),
	Dense(3, kernel_regularizer=regularizers.l2(reg))
	])

	sgd = optimizers.SGD(lr=1e-1, decay=1e-6, momentum=0.9, nesterov=True)
	adam = optimizers.adam(lr=1e-1)
	model.compile(loss="mean_squared_error", optimizer=adam)
	return model

def trainModel(model, states, delay):
	states_delayed = states[delay:]
	states = states[:-delay]
	states_delayed_diff = states_delayed - states
	labels = states_delayed_diff[]
	model.fit(states, labels, epochs=1)

def moveRandom(n):
	data = []
	command = 0
	commandStep = 100
	for _ in range(n):
		action = (2*np.random.randint(2) - 1)
		command += commandStep * action
		command = min(max(command,-2046), 2046)

		start = time()

		x, xpole, ypole = getData()
		state = [x, x-old_x, xpole, xpole-old_xpole, ypole, ypole-old_ypole]
		predicted_state = model.predict(np.array([state]))[0]
		print state
		print (int(200*state[2]+250),int(500-200*state[4]))

		diagram = np.zeros((500,500,3), np.uint8)
		cv2.line(diagram, (int(200*state[0]+150),250), (int(100*state[2] + 200*state[0]+150),int(250-200*state[4])),(255,255,255),5)
		cv2.line(diagram, (int(200*predicted_state[0]+150),250), (int(100*predicted_state[2] + 200*predicted_state[0]+150),int(250-200*predicted_state[4])),(0,255,255),5)
		cv2.imshow("diagram",diagram)

		if cart.enabled:
			data.append(state)


		if x < 0.2:
			command = min(command,0)
		if x > 0.8:
			command = max(command, 0)

		cart.setSpeed(command)


		key = cv2.waitKey(1)
		if key & 0xFF == 32:
			print("toggle enable")
			cart.toggleEnable()
		elif key & 0xFF == ord('q'):
			analyzer.save()
			break
		elif key & 0xFF == ord('r'):
			print("reset")
			reset()

	return data


model = makeModel()


analyzer = ImageAnalyzer(1)

cart = CartCommand("/dev/ttyACM0")

commandqueue = Queue.Queue()

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
	data = moveRandom(100)
	print data
	trainModel(model, data, 4)