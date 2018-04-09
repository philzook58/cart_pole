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


memory = EpisodeParameterMemory(limit=1000, window_length=1)

cem = CEMAgent(model=model, nb_actions=2, memory=memory,
               batch_size=50, nb_steps_warmup=2000, train_interval=50, elite_frac=0.05)
cem.compile()

command_queue = Queue.Queue()


avg = np.array([0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
var = np.array([0.5, 1./50., 2., 1./5., 2.0, 1./5., 2048., 2048., 2048., 2048., 2048.])


def learn(n):
	cem.fit(env, nb_steps=n)


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

		print state


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
