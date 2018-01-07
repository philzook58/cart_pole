import gym

class CustomCartPole(gym.Env):
	def __init__(self):
		self.reset()
		self.dt = 1.0/70
		self.enabled = False
		self.action_space = gym.spaces.Discrete(2)
		self.observation_space = gym.spaces.Discrete(2)

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

	def step(self, action):
		info = {}

		return observation, reward, done, info 
	def render(self):
		pass