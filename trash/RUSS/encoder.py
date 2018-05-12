import serial
import struct
import threading
import time
#http://www.robotshop.com/media/files/pdf/rb-dim-47-datasheet.pdf

mm_per_rev =  22.9 * 3.1459 / 600
rad_per_rev = 2 * 3.1459 / 600
motor_mm_s = 2046 / 1000 # motor moves roughly 1m/s at full tilt?


class CartCommand():
	def __init__(self, port='/dev/ttyACM0', enc_port='/dev/ttyUSB0'):
		self.ser = serial.Serial(port, 115200)  # open serial port
		self.encoder_ser = serial.Serial(enc_port, 115200) 
		self.enabled = False
		self.setSpeed(0)
		#self.cart_position = 0
		#self.pole_position = 0

		self.run = True
		self.end_stop = False
		self.read_thread = threading.Thread(target=update_position_loop, args=(self,))
		time.sleep(0.2)
		self.pole_zero = self.cart_position
		self.cart_zero = self.pole_position

	def __del__(self):
		self.setSpeed(0)
		self.run = False
		self.ser.close()
		self.encoder_ser.close()


	def setSpeed(self, speed):
		# roughly in mm/s
		if not self.enabled:
			return
		speed = int(speed * motor_mm_s)
		speed = max(min(2047, speed),-2046)
		assert -2046 <= speed <= 2047 
		message = "M1: " + str(speed) + "\r\n"
		self.ser.write(message)

	def setRamp(self, ramp):
		if not self.enabled:
			return
		assert -2046 <= ramp <= 2047 
		message = "R1: " + str(int(ramp)) + "\r\n"
		self.ser.write(message)

	def getPosition(self)
		return (self.cart_position - self.cart_zero) * mm_per_rev
	def getPolePosition(self)
		return (self.pole_position - self.pole_zero) * rad_per_rev
	def toggleEnable(self):
		self.setSpeed(0)
		self.enabled = not self.enabled
		if not self.enabled:
			self.setSpeed(0)
	def zero(self):
		self.pole_zero = self.pole_position
		self.cart_zero = self.cart_position
	def stop(self):
		self.setSpeed(0)
	def reset(self):
		#line = self.encoder_ser.readline()
		self.setSpeed(-10)
		while self.end_stop == False:
			pass
		self.setSpeed(0)
		self.cart_zero = self.pole_position

	def update_position_loop(self):
		while self.run:
			line = self.encoder_ser.readline()
			lines = line.split('\t');
			for line in lines:
				if line[0] == 'a':
					self.cart_position = int(line[1:])
				if line[0] == 'b':
					self.pole_position = int(line[1:]) 
				if line[0] == 'c':
					self.end_stop = int(line[1:]) > 0 
	#				if self.end_stop:
	#					self.move(0)

