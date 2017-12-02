import gym
from queue import Queue



env = gym.make('CartPole-v0')
(x, dx, t, dt) = env.reset()

q = Queue()

for _ in range(10):
	obs, reward, done, info = env.step(env.action_space.sample())
	q.put(obs)

for _ in range(1000):
    env.render()

    (x, dx, t, dt) = q.get()

    action = None
    if dt + 5*t < 0:
    	action = 0
    else:
    	action = 1

    obs, reward, done, info = env.step(action)
    q.put(obs)
    print(t)