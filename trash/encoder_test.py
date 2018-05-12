from sabretooth_command import CartCommand
from cart_controller import CartController

from encoder_analyzer import EncoderAnalyzer
import time
import matplotlib.pyplot as plt
import numpy as np


import serial.tools.list_ports
ports = list(serial.tools.list_ports.comports())
print(dir(ports))
for p in ports:
    print(dir(p))
    print(p.device)
    if "Sabertooth" in p.description:
       sabreport = p.device
    else:
       ardPort = p.device

print("Initilizing Commander")
comm = CartCommand(port= sabreport) #"/dev/ttyACM1")
print("Initilizing Analyzer")
analyzer = EncoderAnalyzer(port=ardPort) #"/dev/ttyACM0")
print("Initializing Controller.")
cart = CartController(comm, analyzer)

'''\
while True:
	time.sleep(0.01)
	#print(analyzer.getPolePosition())
	#printanalyzer.getPosition()
	print(analyzer.getPosition(), "   ", analyzer.getPolePosition())

def setPos(self):
'''
print("Starting Zero Routine")
cart.zeroAnalyzer()
cart.goTo(250)

time.sleep(0.5)

speeds = range(240,2040,200)
vels = []

for speed,color in zip(speeds, ['b','r','g']*10):

	times = []

	cart_pos = []
	cart_vel = []
	cart.goTo(200)
	time.sleep(0.1)
	cart.setSpeed(speed)
	start_time = time.time()
	while  cart.analyzer.getCartPosition() < 500: #  time.time() - start_time < 0.5:
		times.append(time.time()-start_time)
		#cart_pos.append(cart.analyzer.getCartVelocity())
		time.sleep(0.01)
		cart_vel.append(cart.analyzer.getCartVelocity())
	cart.setSpeed(0)
	'''
	cart.setSpeed(0)
	while time.time() - start_time < 1.:
		times.append(time.time()-start_time)
		cart_pos.append(cart.analyzer.getCartPosition())
		time.sleep(0.01)
	'''
	plt.scatter(times, cart_vel, c=color)
	vels.append(np.mean(cart_vel))
print(np.polyfit(speeds, vels, 1))
m, b = np.polyfit(speeds, vels, 1)
plt.figure()
plt.scatter(speeds, vels)
plt.plot(speeds, np.array(speeds)*m + b)

plt.show()