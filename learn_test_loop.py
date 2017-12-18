from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers, regularizers
from keras import backend as K

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

class Memory:
	def __init__(self):
		self.states = None
		self.actions = None
		self.next_states = None
		self.newest_states = None
		self.newest_actions = None
		self.newest_next_states = None
	def add(self, new_states, new_actions, new_next_states):
		self.newest_states = np.array(new_states)
		if self.states is None:
			self.states = np.array(new_states)
		else:
			self.states = np.concatenate((self.states, np.array(new_states)), axis=0)

		self.newest_actions = np.array(new_actions)
		if self.actions is None:
			self.actions = np.array(new_actions)
		else:
			self.actions = np.concatenate((self.actions, np.array(new_actions)), axis=0)

		self.newest_next_states = np.array(new_next_states)
		if self.next_states is None:
			self.next_states = np.array(new_next_states)
		else:
			self.next_states = np.concatenate((self.next_states, np.array(new_next_states)), axis=0)
	def __repr__(self):
		return "states: %s\nactions: %s\nnext_states: %s"%(str(np.shape(self.states)),str(np.shape(self.actions)),str(np.shape(self.next_states)))

	def sample(self, n):
		# returns the newest data concatenated with n samples of the old data

		sample_states = sample_actions = sample_next_states = None
		if n > self.states.shape[0]:
			sample_states = self.states
			sample_actions = self.actions
			sample_next_states = self.next_states
		else:
			sample_states = self.states[np.random.choice(self.states.shape[0], n, replace=False), :]
			sample_actions = self.actions[np.random.choice(self.actions.shape[0], n, replace=False)]
			sample_next_states = self.next_states[np.random.choice(self.next_states.shape[0], n, replace=False), :]
		return (np.concatenate((sample_states,self.newest_states), axis=0), 
			np.concatenate((sample_actions,self.newest_actions), axis=0),
			np.concatenate((sample_next_states, self.newest_next_states), axis=0))



######################
# learning functions #
######################

def makeModel():
	reg = 1e-13
	model = Sequential([
	Dense(2, input_shape=(11,), kernel_regularizer=regularizers.l2(reg)),
	Activation('relu'),
	Dense(2, kernel_regularizer=regularizers.l2(reg)),
	Activation('relu'),
	Dense(2, kernel_regularizer=regularizers.l2(reg))
	])

	sgd = optimizers.SGD(lr=1e-1, decay=1e-6, momentum=0.9, nesterov=True)
	adam = optimizers.adam(lr=1e-1)
	model.compile(loss="mean_squared_error", optimizer=sgd)
	return model

model = makeModel()


#model = load_model('1513466579.model')
#model.optimizer.lr.assign(1e-7)

command_queue = Queue.Queue()


avg = np.array([0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
var = np.array([0.5, 1./50., 2., 1./5., 2.0, 1./5., 2048., 2048., 2048., 2048., 2048.])


def learn(n, loss):

	lookahead = 20

	states, actions, next_states = memory.sample(1400)

	for i in range(n):
		gamma = 1.0 - 1.0 / lookahead
		labels = relabel(np.array(actions), np.array(states), np.array(next_states), gamma = gamma)# * (1-gamma)

		history = model.fit(np.array(states), labels, epochs=1)
		print K.eval(model.optimizer.iterations)
		loss += history.history["loss"]
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
	rewards_pole = 0.0 * (states[:,4] + 0.5)**2 #ypole hieght	
	rewards_cart = -2.0 * np.power(states[:,0],2) #xcart pos
	return rewards_cart + rewards_pole


def relabel(actions, states, next_states, gamma=1.0 - 1.0/20 ):
	global left_model, right_model

	rewards = getReward(next_states)

	qs = model.predict(states)
	next_qs = model.predict(next_states)
	action_indices = ((actions + 1)/2).astype(int)
	#print(qs[0:10])
	#print(action_indices.shape)
	#print(np.arange(qs.shape[0]).shape)
	#print(rewards.shape)
	#print(np.max(next_qs, axis=1).shape)
	qs[np.arange(qs.shape[0]),action_indices] = rewards + gamma * np.max(next_qs, axis=1)
	#print(qs[0:10])
	return qs

#####################
# testing functions #
#####################

analyzer = ImageAnalyzer(1)

cart = CartCommand(port="/dev/ttyACM0")

memory = Memory()

def test(n, random_action=False, eps=1.0):
	global states, actions, next_states, command_queue

	command_queue = Queue.Queue()
	cart.toggleEnable()

	current_states = []
	current_actions = []
	current_next_states = []

	command = 0
	commandStep = 500

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


		old_x, old_xpole, old_ypole = x, xpole, ypole

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
	throwaway = 5
	current_next_states = np.array(current_states[throwaway+1:])
	current_states = np.array(current_states[throwaway:-1])
	current_actions = np.array(current_actions[throwaway:-1])

	memory.add(current_states, current_actions, current_next_states)

	resetCart()
	cart.setSpeed(0)
	return current_states, np.sum(getReward(current_next_states))


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
fig2 = plt.figure("state")
fig3 = plt.figure("loss")
fig4 = plt.figure("weights")

plt.ion()
plt.show()
test_rewards = []
loss = []
test_rewards.append(test(100, random_action=True)[1])
file_name = "%d.model"%int(time())
for i in range(100):
	learn(100, loss)
	states, reward = test(600, eps= 0.2 + 2.0/(i+1))
	test_rewards.append(reward)

	fig.clear()
	fig2.clear()
	fig3.clear()
	fig4.clear()
	plt.figure("reward")
	plt.plot(test_rewards)
	plt.figure("state")
	for j in range(states.shape[1]):
		plt.plot(states[:,j], label=str(j))
	plt.legend()

	plt.figure("loss")
	plt.plot(loss)

	plt.figure("weights")
	print model.get_weights()
	plt.hist(np.concatenate([layer.flatten() for layer in model.get_weights()]))
	plt.yscale('log', nonposy='clip')


	plt.pause(.0001)
	model.save(file_name,"w")
