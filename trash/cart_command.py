import serial
import struct


class CartCommand():
	def __init__(self, port='/dev/ttyUSB0'):
		self.ser = serial.Serial(port, 115200)  # open serial port
		self.enabled = False
	def __del__(self):
		self.ser.close()

	def move(self, stepNum, collisionEstimate): 
		if not self.enabled:
			return
		if stepNum < 0:
			dirchar = b'b'
		else:
			dirchar = b'f'
		stepNum = int(abs(stepNum))
		if stepNum * 1.2 < collisionEstimate:
			self._raw_command(dirchar, stepNum)

	def toggleEnable(self):
		self.enabled = not self.enabled

	def _raw_command(self, commandByte, num):
		message = struct.pack('ccBc' , b'^', commandByte, min(255,num), b';')
		self.ser.write(message)
		#self.ser.flush()
	def setMaxSpeed(self, speed):
		self._raw_command(b's', speed)