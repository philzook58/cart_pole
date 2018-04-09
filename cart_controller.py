from sabretooth_command import CartCommand
from encoder_analyzer import EncoderAnalyzer
import threading



class CartController:
	def __init__(self, cart, analyzer):
		self.cart = cart
		self.analyzer = analyzer
		self.speed = 0
		self.cart.enabled = True

		self.mm_per_s_per_dc = 0.5 #determined experimentally
		#self.limit_switch_thread = threading.Thread(target=self._limit_switch_check)
		#self.limit_switch_thread.daemon = True
		#self.limit_switch_thread.start()

	def _limit_switch_check(self):
		while True:
			if (self.speed > 0) and self.analyzer.end_stop_high:
				self.cart.setSpeed(0)
				print("high limit switch")
			if (self.speed < 0) and self.analyzer.end_stop_low:
				self.cart.setSpeed(0)
				print("low limit switch")

	def setSpeed(self, speed):
		self.speed = max(-2046, min(speed, 2047)) # clamp to correct range
		self.cart.setSpeed(self.speed)

	def setSpeedMmPerS(self, speed):
		speed *= self.mm_per_s_per_dc
		if speed < 0:
			speed -= 100
		else:
			speed += 100
		self.speed = max(-2046, min(speed, 2047)) # clamp to correct range
		self.cart.setSpeed(self.speed)

	def zeroAnalyzer(self):
		while not self.analyzer.checkEndStopHigh():
			self.cart.setSpeed(700)
		self.analyzer.setMax()
		while not self.analyzer.checkEndStopLow():
			self.cart.setSpeed(-700)
		self.cart.setSpeed(0)
		self.analyzer.setZero()

	def goTo(self, target_x):
		print(self.analyzer.getCartPosition(), self.analyzer.getPolePosition())
		while (self.analyzer.getCartPosition() - target_x) > 25:
			self.cart.setSpeed(-1000)
		while (self.analyzer.getCartPosition() - target_x) < -25:
			self.cart.setSpeed(1000)
		self.cart.setSpeed(0)

