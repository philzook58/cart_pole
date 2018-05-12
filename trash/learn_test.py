from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
import numpy as np
import pickle
import sys

import cv2
from sabretooth_command import CartCommand
from image_analyzer_pseye import ImageAnalyzer
from time import time, sleep
import Queue
from keras.models import load_model


def reset_data():
	global states, next_states, command_queue, actions
	states = []
	next_states = []
	actions = []
	command_queue = Queue.Queue()



######################
# learning functions #
######################

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

left_model = makeModel()
right_model = makeModel()

states = []
next_states = []
actions = []
command_queue = Queue.Queue()


avg = np.array([0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
var = np.array([0.5, 1./50., 2., 1./50., 2.0, 1./50., 2048., 2048., 2048., 2048., 2048.])


def learn(n):

	left_states = np.array([states[i] for i in range(len(states)) if actions[i] == -1])
	left_next_states =  np.array([next_states[i] for i in range(len(states)) if actions[i] == -1])

	right_states = np.array([states[i] for i in range(len(states)) if actions[i] == 1])
	right_next_states =  np.array([next_states[i] for i in range(len(states)) if actions[i] == 1])

	#left_states = normalize(left_states)	
	#left_next_states = normalize(left_next_states)
	#right_states = normalize(right_states)
	#right_next_states = normalize(right_next_states)

	lookahead = 300

	for i in range(n):
		gamma = 1.0 - 1.0 / lookahead
		leftQ = relabel(left_next_states, gamma = gamma)# * (1-gamma)
		rightQ = relabel(right_next_states, gamma = gamma)# * (1-gamma)

		left_model.fit(left_states, leftQ, epochs=1)
		right_model.fit(right_states, rightQ, epochs=1)
'''
def calc_norms(data):
	#print(data.shape)

	avg = np.mean(data, axis=0).reshape((1,-1))
	var = np.std(data,axis=0).reshape((1,-1))
	#print(avg.shape)
	return avg, var
'''
def normalize(data):
	#print (data - avg)/var
	global avg, var
	return (data - avg)/var


def relabel(next_states, gamma=1.0 - 1.0/20 ):
	global left_model, right_model
	rewards_pole = (next_states[:,4].reshape((-1,1)) + 0.5)**2 #ypole hieght	
	rewards_cart = 0.1 * np.power(next_states[:,0].reshape((-1,1)),2) #xcart pos
	maxQs = np.maximum(left_model.predict(next_states), right_model.predict(next_states))
	#print(maxQs.shape)
	#print(rewards.shape)
	labels = rewards_pole + (gamma*maxQs) - rewards_cart
	return labels

def makeModel():
	model = Sequential([
	Dense(10, input_shape=(11,)),
	Activation('relu'),
	Dense(10),
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

def test(n, random_action=False, eps=1.0):
	global states, actions, next_states, command_queue
	resetCart()

	command_queue = Queue.Queue()
	cart.toggleEnable()

	current_states = []
	current_actions = []
	current_next_states = []

	command = 0
	commandStep = 100

	for i in range(5):
		command_queue.put(0)

	old_x, old_xpole, old_ypole = getData()

	for _ in range(n):

		start = time()

		x, xpole, ypole = getData()
		state = normalize(np.array([x, x-old_x, xpole, xpole-old_xpole, ypole, ypole-old_ypole] + list(command_queue.queue)))

		if np.random.rand() < eps or random_action:
			action = (2*np.random.randint(2) - 1)
		else:
			q_left = left_model.predict(state.reshape((1,-1)))
			q_right = right_model.predict(state.reshape((1,-1)))

			print q_left,q_right
			action = 0
			if q_left > q_right:
				action = -1
			else:
				action = 1

		current_states.append(state)
		current_actions.append(action)


		command += commandStep * action
		command = min(max(command,-2046), 2046)


		if x < 0.35:
			command = min(command, -500)
		if x > 0.65:
			command = max(command, 500) 

		cart.setSpeed(command)
		command_queue.put(command)
		command_queue.get()




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

	current_next_states = current_states[1:]
	current_states = current_states[0:-1]	
	current_actions = current_actions[0:-1]

	states += current_states
	actions += current_actions
	next_states += current_next_states
	cart.setSpeed(0)



def resetCart():	
	x, dx, theta, dtheta = analyzer.analyzeFrame()
	cart.enabled = True
	while not 0.4 < x < 0.6:
		x, dx, theta, dtheta = analyzer.analyzeFrame()

		command = 1000 * np.sign(x-0.5)
		command = min(max(command,-2046), 2046)

		cart.setSpeed(command)
		cv2.waitKey(1)

	cart.setSpeed(0)
	sleep(0.3)
	cart.enabled = False

def getData():	
	x, dx, theta, dtheta = analyzer.analyzeFrame()
	xpole = np.cos(theta)
	ypole = np.sin(theta)
	return x, xpole, ypole


test(500, random_action=True)
for i in range(100):
	learn(20)
	test(500, eps= 1.0/(i+1))