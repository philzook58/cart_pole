from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from keras.models import load_model
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt

import cv2
#from sabretooth_command import CartCommand
#from image_analyzer_pseye import ImageAnalyzer
from time import time, sleep
#from multiprocessing import Queue
import queue

from scipy.integrate import odeint 

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from keras.models import load_model

class CartCommandAndImage():
	def __init__(self):
		self.reset()
		self.dt = 1.0/70
		self.enabled = False
	def derivs(self,state, F):
		responseTime = self.dt
		alpha = 1/ responseTime
		maxVel = 0.5
		beta = alpha * maxVel / 2046


		x, xdot, theta, thetadot = state
		dx = xdot
		dxdot = beta * F - alpha * xdot 
		dtheta = thetadot 
		dthetadot = 2 / 3 / 0.4 * (dxdot * np.sin(theta) - 20 * np.cos(theta))

		return np.array([dx, dxdot, dtheta, dthetadot])
	def setSpeed(self, speed):
		stateVec = np.array([self.x, self.xdot, self.theta, self.thetadot])
		states = odeint(lambda state, t: self.derivs(state, speed), stateVec, np.array([0,self.dt]))
		print(states)
		self.x, self.xdot, self.theta, self.thetadot = states[-1] #list(states[-1])

	def analyzeFrame(self):
		return self.x, self.xdot * self.dt, self.theta , self.thetadot* self.dt
	def reset(self):
		self.x = 0.5
		self.xdot = 0
		self.theta = -np.pi/2
		self.thetadot = 0
	def toggleEnable(self):
		self.reset()









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







def learn(model, n, lr=1e-8):
	model.lr.set_value(lr)
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

#  Data type episodes = [{'action': np.array, state: np.array, 'model': }] 
#make a class?
episodes = []


def normalize(data): #This is expecting data to be nparray
	avg = np.array([0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
	var = np.array([0.5, 1./50., 2., 1./50., 2.0, 1./50., 2048., 2048., 2048., 2048., 2048.])
	return (data - avg)/var

def getStepReward(states):
	rewards_pole = (states[:,4].reshape((-1,1)) + 0.5)**2 #ypole hieght	
	rewards_cart = 0.1 * np.power(states[:,0].reshape((-1,1)),2) #xcart pos
	return rewards_cart + rewards_pole


def relabel(actions, states, next_states, gamma=1.0 - 1.0/20 ):
	rewards = getStepReward(next_states)

	qs = model.predict(states)
	next_qs = qs[1:]
	this_qs = qs[:-1]
	#next_qs = model.predict(next_states)
	action_indices = ((actions + 1)/2).astype(int)

	qs[:,action_indices] = rewards + gamma * np.max(next_qs, axis=1)

	return qs

#####################
# testing functions #
#####################

#analyzer = ImageAnalyzer()

cart = CartCommandAndImage()
analyzer = cart

def test(n, random_action=False, eps=1.0):

	command_queue = queue.Queue()
	cart.toggleEnable()

	current_states = []
	current_actions = []

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
			action = np.random.randint(2)
		else:
			q = model.predict(state.reshape((1,-1))).reshape(2)

			print(q)
			action = np.argmax(q)


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

	resetCart()
	cart.setSpeed(0)

	return  current_states, current_actions

# should be in cart command kind of?
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

# Is this dumb?
def getData():	
	x, dx, theta, dtheta = analyzer.analyzeFrame()
	xpole = np.cos(theta)
	ypole = np.sin(theta)
	return x, xpole, ypole


fig = plt.figure("reward")
plt.ion()
plt.show()
test_rewards = []
episodes = []
states, actions = test(100, random_action=True)
reward = [np.sum(getStepReward(states))]
episodes.append({'states':states, 'actions':actions})
for i in range(100):
	learn(30)
	states, actions = test(100, eps= 0.2 + 2.0/(i+1))
	rewards.append(np.sum(getStepReward(states)))
	episodes.append({'states':states, 'actions':actions})
	fig.clear()
	plt.plot(test_rewards)
	plt.pause(.0001)