class CartPole:
	def __init__(self, motor, pos_analyzer, angle_analyzer, end_analyzer):
		self.motor = motor
		self.pos_analyzer = pos_analyzer
		self.angle_analyzer = angle_analyzer
		self.end_analyzer = end_analyzer
		self.speed = 0
		self.motor.enabled = True

		self.mm_per_s_per_dc = 0.5 

	def setSpeed(self, speed):
		self.speed = max(-2046, min(speed, 2047)) # clamp to correct range
		self.motor.setSpeed(self.speed)

	def setSpeedMmPerS(self, speed):
		speed *= self.mm_per_s_per_dc
		if speed < 0:
			speed -= 100
		else:
			speed += 100
		self.speed = max(-2046, min(speed, 2047)) # clamp to correct range
		self.motor.setSpeed(self.speed)

	def zeroAngleAnalyzer(self):
		self.angle_analyzer.setAngleZero()

	def zeroPosAnalyzer(self):
		while not self.end_analyzer.getEndStopHigh():
			self.angle_analyzer.updateState()
			self.motor.setSpeed(700)
		self.pos_analyzer.setPosMax()
		while not self.end_analyzer.getEndStopLow():
			self.angle_analyzer.updateState()
			self.motor.setSpeed(-700)
		self.motor.setSpeed(0)
		self.pos_analyzer.setPosZero()

	def goTo(self, target_x):
		while (self.pos_analyzer.getPos() - target_x) > .1:
			self.angle_analyzer.updateState()
			self.motor.setSpeed(-1000)
		while (self.pos_analyzer.getPos() - target_x) < -.1:
			self.angle_analyzer.updateState()
			self.motor.setSpeed(1000)
		self.motor.setSpeed(0)


	def getState(self):
		self.pos_analyzer.updateState()
		self.angle_analyzer.updateState()
		return (
			self.pos_analyzer.getPosMm(update=False), 
			self.pos_analyzer.getVelMm(update=False), 
			self.angle_analyzer.getAngle(update=False), 
			self.angle_analyzer.getAngleVel(update=False))