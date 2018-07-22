import serial

class MotorControl:
	def __init__(self):
		self.enabled = False

	def __del__(self):
		self.setSpeed(0)

	def setSpeed(self, speed):
		assert False, "setSpeed not implemented"

	def enable(self):
		self.setSpeed(0)
		self.enabled = True
		if not self.enabled:
			self.setSpeed(0)

	def disable(self):
		self.setSpeed(0)
		self.enabled = False
		if not self.enabled:
			self.setSpeed(0)


class SabreControl(MotorControl):
	def __init__(self, port='/dev/ttyACM1'):
		super().__init__()
		self.ser = serial.Serial(port, 115200)  # open serial port
		self.setSpeed(0)
	def __del__(self):
		super().__del__()
		self.ser.close()


	def setSpeed(self, speed):
		if self.enabled:
			assert -2046 <= speed <= 2047 
			message = "M1: " + str(int(speed)) + "\r\n"
			self.ser.write(message.encode('utf-8'))
