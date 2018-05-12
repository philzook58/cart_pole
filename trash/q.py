from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
import numpy as np
import pickle
import sys

data = pickle.load(open(sys.argv[1],'r'))


def arrayify(data):
	states = [entry['state'] for entry in data]
	actions = [entry['action'] for entry in data]
	newstates = [entry['next_state'] for entry in data]
	return {"states": np.array(states), "actions": np.array(actions), "next_states": np.array(newstates)}

left_data = [entry for entry in data if entry['action'] == -1]
left_data = arrayify(left_data)
right_data = [entry for entry in data if entry['action'] == 1]
right_data = arrayify(right_data)

avg = var = None

def calc_norms(data):
	#print(data.shape)

	avg = np.mean(data, axis=0).reshape((1,-1))
	var = np.std(data,axis=0).reshape((1,-1))
	#print(avg.shape)
	return avg, var

def normalize( data , avg, var):
	return (data - avg)/var

avg, var = calc_norms(left_data['states'])

right_data['next_states'] = normalize(right_data['next_states'], avg, var)
right_data['states'] = normalize(right_data['states'] , avg, var)
left_data['next_states'] = normalize(left_data['next_states'] , avg, var)
left_data['states'] = normalize(left_data['states'] , avg, var)


def relabel(data, gamma=1.0 - 1.0/20 ):
	global left_model, right_model
	next_states = data["next_states"]
	rewards_pole= next_states[:,4].reshape((-1,1)) #ypole hieght
	rewards_cart= next_states[:,0].reshape((-1,1)) - 0.5 #ypole hieght
	maxQs = np.maximum(left_model.predict(next_states), right_model.predict(next_states))
	#print(maxQs.shape)
	#print(rewards.shape)
	labels = rewards_pole +  + gamma*maxQs
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

left_model = makeModel()
right_model = makeModel()

#for lookahead in range(20,100):
lookahead = 100
for i in range(1000):
	print("lookahead: ",lookahead)
	gamma = 1.0 - 1.0 / lookahead
	leftQ = relabel(left_data, gamma = gamma)# * (1-gamma)
	rightQ = relabel(right_data, gamma = gamma)# * (1-gamma)
	#print(leftQ)
	#print(left_data['states'])
	left_model.fit(left_data['states'], leftQ, epochs=1)
	right_model.fit(right_data['states'], rightQ, epochs=1)

pickle.dump({"avg":avg, "var":var}, open(sys.argv[1][:-2] + ".norm","w"))
left_model.save(sys.argv[1][:-2] + ".left")
right_model.save(sys.argv[1][:-2] + ".right")