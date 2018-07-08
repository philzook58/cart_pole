import threading
import serial
import time
import numpy as np


mm_per_rev =  -25.0 * 3.1459 / 1200. / 2.
rad_per_rev = 2 * 3.1459 / 1200. / 2.#turns out we're getting 1200 per revolution
motor_mm_s = 2046. / 1000. # motor moves roughly 1m/s at full tilt?


class EncoderAnalyzer():
	def __init__(self, port='/dev/ttyACM0'):
		self.ser = serial.Serial(port, 500000, timeout=1, parity=serial.PARITY_EVEN, stopbits=serial.STOPBITS_ONE)  
		print("Initialized Analyzer Serial")
		self.ser.flushInput()
		self.cart_position = 0
		self.cart_position_last = 0
		self.pole_position = 0
		self.pole_position_last = 0
		self.time = 0.1
		self.time_last = 0
		self.cart_zero = 0
		self.cart_max = 0
		self.pole_zero = 0
		self.end_stop_low = False
		self.end_stop_high = False		

		self.time_history = []
		self.pos_history = []

		self.interrupt_history = []

		self.run = True
		print("Starting Read Thread")
		#self.update_position_thread = threading.Thread(target=self._update_position)
		#self.update_position_thread.daemon = True
		
		#self.update_position_thread.start()

	def setZero(self):
		self.cart_zero = self.cart_position

	def setMax(self):
		self.cart_max = self.cart_position

	def getCartPosition(self):
		self._update_position()
		return -(self.cart_position - self.cart_zero) * mm_per_rev

	def checkEndStopHigh(self):
		self._update_position()
		return self.end_stop_high

	def checkEndStopLow(self):
		self._update_position()
		return self.end_stop_low

	def getCartVelocity(self):
		self._update_position()
		time.sleep(1./100)
		self._update_position()
		dt = self.time - self.time_last
		return (self.cart_position - self.cart_position_last) * mm_per_rev / dt

	def getPolePosition(self):
		self._update_position()
		return (self.pole_position - self.pole_zero) * rad_per_rev

	def getPoleVelocity(self):
		self._update_position()
		time.sleep(1./100)
		self._update_position()
		dt = self.time - self.time_last
		return ((self.pole_position - self.pole_position_last)) * rad_per_rev / dt


	def getState(self):
		self._update_position()
		time.sleep(1./500)
		self._update_position()
		dt = self.time - self.time_last
		return np.array([
			-(self.cart_position - self.cart_zero) * mm_per_rev, 
			-(self.cart_position - self.cart_position_last) * mm_per_rev / dt, 
			(self.pole_position - self.pole_zero) * rad_per_rev, 
			((self.pole_position - self.pole_position_last)) * rad_per_rev / dt
			])

	def getMax(self):
		return -(self.cart_max - self.cart_zero) * mm_per_rev

	def _update_position(self):
		#while True:
		#time.sleep(0.001)
		self.ser.write(b'a')
		line = self.ser.readline()
		#self.ser.reset_input_buffer
		#print(line)
		try:
			lines = line.split(b'\t')
			if len(lines) == 4:
				self.cart_position_last = self.cart_position
				self.cart_position = int(lines[0])

				self.pole_position_last = self.pole_position
				self.pole_position = int(lines[1]) 



				#self.time_last, self.time = self.time, time.time()

				self.end_stop_high = (int(lines[3]) > 0)
				self.end_stop_low = (int(lines[2]) > 0) 
				self.time_last = self.time
				self.time = time.time()

				#self.pos_history.append(self.cart_position)
				#self.time_history.append(self.time)
			else:
				print("Length of arduino message not 4")
		except ValueError:
			print("Bad Serial Response")

	def __del__(self):
		self.run = False
		self.ser.close()
