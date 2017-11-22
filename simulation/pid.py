import gym


env = gym.make('CartPole-v0')
(x, dx, t, dt) = env.reset()

env.step(env.action_space.sample())
for _ in range(1000):
    env.render()

    action = None
    if dt + t < 0:
    	action = 0
    else:
    	action = 1

    (x, dx, t, dt), reward, done, info = env.step(action)
    print(t)