import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from custom_cartpole_xy_predict import CartPoleEnv

ENV_NAME = 'CartPole-v0'


# Get the environment and extract the number of actions.
env = CartPoleEnv() #gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
print((1,) + env.observation_space.shape)
pmodel = Sequential()
pmodel.add(Flatten(input_shape=(1,) + env.observation_space.shape))
pmodel.add(Dense(16))
pmodel.add(Activation('relu'))
pmodel.add(Dense(16))
pmodel.add(Activation('relu'))
pmodel.add(Dense(16))
pmodel.add(Activation('relu'))
pmodel.add(Dense(env.observation_space.shape[0]))
pmodel.add(Activation('linear'))
print(pmodel.summary())

pmodel.compile(optimizer='rmsprop',
              loss='mse')

delay = 5

def predictor_data():
    episode_count = 10
    #delay = 5
    obs = []
    for i in range(episode_count):
        ob = env.reset()
        obs.append([ob])
        while True:
            action = env.action_space.sample()
            ob, reward, done, _ = env.step(action)
            obs[i].append(ob)
            pob = pmodel.predict(ob.reshape((-1,1,5))).flatten()
           #env._render(angle=np.arctan2(pob[2]+ob[2], pob[3]+ob[3]))
            if done:
                break
    inobs = [np.array(obs1[delay:]) for obs1 in obs]
    
    inobs = np.vstack(inobs).reshape((-1,1,5))
    outobs = np.vstack([ np.array(obs1[:-delay])  for obs1 in obs]).reshape((-1,1,5))
    print(inobs)
    diffobs = outobs - inobs
    return inobs, diffobs
def show():
        ob = env.reset()
        pobs = []
        obs = []
        for i in range(delay):
            pobs.append(np.zeros(5))
            obs.append(np.zeros(5))

        while True:
            action = env.action_space.sample()
            ob, reward, done, _ = env.step(action)
            obs.append(ob)
            #obs[i].append(ob)
            pob = pmodel.predict(ob.reshape((-1,1,5))).flatten()
            pobs.append(pob)
            pob = pobs.pop(0)
            oldob = obs.pop(0)
            env._render(angle=-1 * np.arctan2(oldob[3], oldob[2]), pangle=-1 * np.arctan2(-pob[3]+oldob[3], -pob[2]+oldob[2]))
            if done:
                break


for i in range (30):
    inobs, diffobs = predictor_data()
    # Train the model, iterating on the data in batches of 32 samples
    pmodel.fit(inobs, diffobs.reshape((-1,5)), epochs=10, batch_size=32)
    show()

