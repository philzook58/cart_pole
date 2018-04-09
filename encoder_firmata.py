import time
from pyfirmata import Arduino, util

mm_per_rev =  25.0 * 3.1459 / 1200.
rad_per_rev = 2 * 3.1459 / 1200. #turns out we're getting 1200 per revolution
motor_mm_s = 2046. / 1000. # motor moves roughly 1m/s at full tilt?

board = Arduino('/dev/ttyACM0')
it = util.Iterator(board)
it.run()
in1 = board.get_pin('d:8:i')


while True:
	print(in1.read())

'''
class EncoderAnalyzer():
	def __init__(self, port='/dev/ttyACM0'):
		self.board = Arduino(port)
		self.board.
		'''