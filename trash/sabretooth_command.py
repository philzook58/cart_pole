import serial
import struct

#http://www.robotshop.com/media/files/pdf/rb-dim-47-datasheet.pdf

class CartCommand():
	def __init__(self, port='/dev/ttyACM0'):
		self.ser = serial.Serial(port, 115200)  # open serial port
		self.enabled = False



		self.setSpeed(0)
	def __del__(self):
		self.setSpeed(0)
		self.ser.close()

	def setSpeed(self, speed):
		if not self.enabled:
			return
		assert -2046 <= speed <= 2047 
		message = "M1: " + str(int(speed)) + "\r\n"
		self.ser.write(message)

	def setRamp(self, ramp):
		if not self.enabled:
			return
		assert -2046 <= ramp <= 2047 
		message = "R1: " + str(int(ramp)) + "\r\n"
		self.ser.write(message)

	def toggleEnable(self):
		self.setSpeed(0)
		self.enabled = not self.enabled
		if not self.enabled:
			self.setSpeed(0)