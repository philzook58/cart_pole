from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
import numpy as np
import pickle
import sys

import cv2
from sabretooth_command import CartCommand
from image_analyzer_pseye import ImageAnalyzer
from time import time
import Queue
from keras.models import load_model


def reset_data():
	states = []
	next_states = []
	actions = []


######################
# learning functions #
######################

left_model = makeModel()
right_model = makeModel()

states = []
next_states = []
actions = []

avg = var = None

def learn(n):

	left_states = [states[i] for i in range(len(states)) if action[i] == -1]
	left_next_states = [next_states[i] for i in range(len(states)) if action[i] == -1]

	right_states = [states[i] for i in range(len(states)) if action[i] == 1]
	right_next_states = [next_states[i] for i in range(len(states)) if action[i] == 1]

	
	avg, var = calc_norms(left_data['states'])

	left_states = normalize(left_states, avg, var)	
	left_next_states = normalize(left_next_states, avg, var)
	right_states = normalize(right_states, avg, var)
	right_next_states = normalize(right_next_states, avg, var)


	for i in range(n):
		gamma = 1.0 - 1.0 / lookahead
		leftQ = relabel(left_next_states, gamma = gamma)# * (1-gamma)
		rightQ = relabel(right_next_states, gamma = gamma)# * (1-gamma)

		left_model.fit(left_states, leftQ, epochs=1)
		right_model.fit(right_states, rightQ, epochs=1)

def calc_norms(data):
	#print(data.shape)

	avg = np.mean(data, axis=0).reshape((1,-1))
	var = np.std(data,axis=0).reshape((1,-1))
	#print(avg.shape)
	return avg, var

def normalize( data , avg, var):
	return (data - avg)/var


def relabel(next_states, gamma=1.0 - 1.0/20 ):
	global left_model, right_model
	rewards_pole= next_states[:,4].reshape((-1,1)) #ypole hieght
	rewards_cart= next_states[:,0].reshape((-1,1)) - 0.5 #ypole hieght
	maxQs = np.maximum(left_model.predict(next_states), right_model.predict(next_states))
	#print(maxQs.shape)
	#print(rewards.shape)
	labels = rewards_pole + gamma*maxQs
	return labels

def makeModel():
	model = Sequential([
	Dense(5, input_shape=(11,)),
	Activation('relu'),
	Dense(5),
	Activation('relu'),
	Dense(1)
	])

	sgd = optimizers.SGD(lr=1e-5, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss="mean_squared_error", optimizer=sgd)
	return model

#####################
# testing functions #
#####################

analyzer = ImageAnalyzer()

cart = CartCommand(port="/dev/ttyACM0")

command_queue = Queue.Queue()

for i in range(5):
	command_queue.put(0)

def test(n):
	reset_data()
	for _ in range(n):

		start = time()

		x, xpole, ypole = getData()
		state = [x, x-old_x, xpole, xpole-old_xpole, ypole, ypole-old_ypole] + list(commandqueue.queue)

		q_left = left_model.predict((state - norm['avg']) / norm['var'])
		q_right = right_model.predict((state - norm['avg']) / norm['var'])

		action = 0
		if q_left > q_right:
			action = -1
		else:
			action = 1

		states.append(state)
		actions.append(action)
		try:
			next_states[-2] = state
		except IndexError:
			pass

		command += commandStep * action
		command = min(max(command,-2046), 2046)


		print("x%f"%x)

		if x < 0.35:
			command = min(command,0)
		if x > 0.65:
			command = max(command, 0)

		cart.setSpeed(command)
		commandqueue.put(command)
		commandqueue.get()
		print(commandqueue.queue)




		key = cv2.waitKey(1)
		if key & 0xFF == 32:
			print("toggle enable")
			old_data += 
			data = []
			cart.toggleEnable()
		elif key & 0xFF == ord('q'):
			analyzer.save()
			break
		elif key & 0xFF == ord('r'):
			print("reset")
			reset()

	states = states[0:-1]	
	actions = actions[0:-1]


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


