from pyfirmata import Arduino, util
import time

board = Arduino('/dev/ttyUSB0')
it = util.Iterator(board)
it.start()

#board.digital[13].write(1)

#board.analog[0].enable_reporting()
analog_0 = board.get_pin('a:0:i')

while True:
	print(analog_0.read())
	time.sleep(0.1)