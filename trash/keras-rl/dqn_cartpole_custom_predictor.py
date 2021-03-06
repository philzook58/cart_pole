import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from custom_cartpole_xy_delay_predict import CartPoleEnv

ENV_NAME = 'CartPole-v0'

delay = 1


# Get the environment and extract the number of actions.
env = CartPoleEnv() #gym.make(ENV_NAME)

env.buffer_size = delay
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

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

env.predictor_network = pmodel
# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy, enable_dueling_network=True, dueling_type='avg')
#dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
def predictor_data():
    episode_count = 10
    #delay = 5
    obs = []
    for i in range(episode_count):
        ob = env.reset()
        obs.append([ob])
        while True:
            #action = env.action_space.sample()
            #if len(obs[i]) > delay:
            #    delayedob = obs[i][-delay]
            #else:
            #    delayedob = obs[i][0]
            #It is already delayed in the env
            #pob = pmodel.predict(ob.reshape((-1,1,5))).flatten()
            action = dqn.forward(ob)
            ob, reward, done, _ = env.step(action)
            obs[i].append(env.buffer[-1])
            #pob = pmodel.predict(ob.reshape((-1,1,5))).flatten()
            #env._render(angle=np.arctan2(pob[2]+ob[2], pob[3]+ob[3]))
            if done:
                break
    actualobs = [np.array(obs1[delay:]) for obs1 in obs]
    
    actualobs = np.vstack(actualobs).reshape((-1,1,5))
    delayedobs = np.vstack([ np.array(obs1[:-delay])  for obs1 in obs]).reshape((-1,1,5))
    #print(inobs)
    diffobs = actualobs - delayedobs
    return delayedobs, diffobs

for i in range (10):
    env.return_predicted = True
    delayedobs, diffobs = predictor_data()
    env.return_predicted = True
    pmodel.fit(delayedobs, diffobs.reshape((-1,5)), epochs=10, batch_size=32)
    dqn.fit(env, nb_steps=4096, visualize=False, verbose=2)
    #inobs, diffobs = predictor_data()
    dqn.test(env, nb_episodes=5, visualize=True)

    # Train the model, iterating on the data in batches of 32 samples

    #show()




# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)