from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
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


def reset_data():
	global states, next_states, command_queue, actions
	states = []
	next_states = []
	actions = []
	command_queue = Queue.Queue()


class Memory:
	def __init__(self,state_size):
		self.states = None
		self.actions = None
		self.next_states = None
	def add(new_states, new_actions, new_next_states):
		if self.states is None:
			self.states = np.array(new_states)
		else:
			self.states = np.concatenate(self.states, np.array(new_states))

		if self.actions is None:
			self.actions = np.array(new_actions)
		else:
			self.actions = np.concatenate(self.actions, np.array(new_actions))

		if self.next_states is None:
			self.next_states = np.array(new_next_states)
		else:
			self.next_states = np.concatenate(self.next_states, np.array(new_next_states))
	def __repr__:
		return "states: %s\nactions: %s\nnext_states: %s"%(str(self.states),str(self.actions),str(self.next_states))


	def resample(self, fraction):
		self.states = self.states[np.random.rand(np.shape(self.states)[0])<fraction,:]

print Memory(11).states


######################
# learning functions #
######################

def makeModel():
	model = Sequential([
	Dense(26, input_shape=(11,)),
	Activation('relu'),
	Dense(26),
	Activation('relu'),
	Dense(2)
	])

	sgd = optimizers.SGD(lr=1e-8, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss="mean_squared_error", optimizer=sgd)
	return model

model = makeModel()

states = np.array([])
next_states = np.array([])
actions = np.array([])
command_queue = Queue.Queue()


avg = np.array([0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
var = np.array([0.5, 1./50., 2., 1./50., 2.0, 1./50., 2048., 2048., 2048., 2048., 2048.])


def learn(n):

	lookahead = 100

	for i in range(n):
		gamma = 1.0 - 1.0 / lookahead
		labels = relabel(np.array(actions), np.array(states), np.array(next_states), gamma = gamma)# * (1-gamma)

		model.fit(np.array(states), labels, epochs=1)
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

def getReward(states):
	rewards_pole = (states[:,4].reshape((-1,1)) + 0.5)**2 #ypole hieght	
	rewards_cart = 0.1 * np.power(states[:,0].reshape((-1,1)),2) #xcart pos
	return rewards_cart + rewards_pole


def relabel(actions, states, next_states, gamma=1.0 - 1.0/20 ):
	global left_model, right_model

	print(np.shape(next_states))
	rewards = getReward(next_states)

	qs = model.predict(states)
	next_qs = model.predict(next_states)
	action_indices = ((actions + 1)/2).astype(int)

	qs[:,action_indices] = rewards + gamma * np.max(next_qs, axis=1)

	return qs

#####################
# testing functions #
#####################

analyzer = ImageAnalyzer(1)

cart = CartCommand(port="/dev/ttyACM0")

def test(n, random_action=False, eps=1.0):
	global states, actions, next_states, command_queue

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
			q = model.predict(state.reshape((1,-1)))

			print q
			action = 0
			if q[0][0] > q[0][1]:
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

	current_next_states = np.array(current_states[1:])
	current_states = np.array(current_states[0:-1])
	current_actions = np.array(current_actions[0:-1])

	print np.shape(states)
	if np.shape(states)[0] > 0:
		states = states[np.random.rand(np.shape(states)[0])<0.8,:]
		states = np.append(states, np.array(current_states),axis=0)
	else:
		states = current_states

	if np.shape(actions)[0] > 0:
		actions = actions[np.random.rand(np.shape(actions)[0])<0.8]
		actions = np.append(actions, np.array(current_actions),axis=0)
	else:
		actions = current_actions


	if np.shape(next_states)[0] > 0:
		next_states = next_states[np.random.rand(np.shape(next_states)[0])<0.8,:]
		next_states = np.append(next_states, np.array(current_next_states),axis=0)
	else:
		next_states = current_next_states

	resetCart()
	cart.setSpeed(0)
	print np.shape(next_states)
	return np.sum(getReward(current_next_states))


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


fig = plt.figure("reward")
plt.ion()
plt.show()
test_rewards = []
test_rewards.append(test(100, random_action=True))
for i in range(100):
	learn(30)
	test_rewards.append(test(300, eps= 0.2 + 2.0/(i+1)))
	fig.clear()
	plt.plot(test_rewards)
	plt.pause(.0001)