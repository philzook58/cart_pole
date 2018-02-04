import logging
import math
import gym
from gym import spaces
import numpy as np

from sabretooth_command import CartCommand
from image_analyzer_pseye import ImageAnalyzer
import cv2

logger = logging.getLogger(__name__)

class CartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, cartport, imageport):
        self.port = port
        self.analyzer = ImageAnalyzer(imageport)
        self.cart = CartCommand(port=cartport)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(np.array([0.,-50.,0.,-50.,-1.,-50.]), np.array([1.,50.,1.,50.,1.,50.]))

        self.state = self._getState()
        self.last_state = self._getState()


    def _step(self, action):
        if action == self.action_space[0]:
            d_command = 1.
        else:
            d_command = -1.

        command += commandStep * d_command
        command = min(max(command,-2046), 2046)


        if x < 0.35:
            command = min(command, -500)
        if x > 0.65:
            command = max(command, 500) 

        self.cart.setSpeed(command)

        self.last_state = self.state
        self.state = self._getState()
        reward = self._getReward(self.state)
        done = False

        return np.array(self.state), reward, done, {}

    def _reset(self):
        x, dx, theta, dtheta = self.analyzer.analyzeFrame()
        cart.enabled = True
        while not 0.4 < x < 0.6:
            x, dx, theta, dtheta = self.analyzer.analyzeFrame()

            command = 1000 * np.sign(x-0.5)
            command = min(max(command,-2046), 2046)

            self.cart.setSpeed(command)
            cv2.waitKey(1)

        self.cart.setSpeed(0)
        sleep(0.3)
        self.cart.enabled = False

    def _getData(self):  
        x, dx, theta, dtheta = self.analyzer.analyzeFrame()
        xpole = np.cos(theta)
        ypole = np.sin(theta)
        return x, xpole, ypole

    def _getState():
        x, xpole, ypole = self._getData
        if not old_state is None:
            state = [x, x-old_state[0], xpole, xpole - old_state[2], ypole, ypole - old_state[4]]
        else: 
            state = [x, 0, xpole, 0, ypole, 0]
        return state

    def _getReward(self, state):
        rewards_pole = 0.0 * (state[:,4] + 0.5)**2 #ypole hieght   
        rewards_cart = -2.0 * np.power(state[:,0],2) #xcart pos
        return rewards_cart + rewards_pole


    def _render(self, mode='human', close=False):
        pass