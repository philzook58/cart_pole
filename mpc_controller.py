import time
import numpy as np
import serial.tools.list_ports
import scipy.linalg as linalg





from analyzer import EncoderAnalyzer, ImageAnalyzer
from motor_control import SabreControl
from cart_pole import CartPole
import serial.tools.list_ports
from mpc import MPC

mpc = MPC(0.5,0,0,0)

ports = list(serial.tools.list_ports.comports())
print(ports)
for p in ports:
    print(p)
    if p[2] == "USB VID:PID=268b:0201 SNR=160045F45953":
       sabre_port = p[0]
    elif p[2] == "USB VID:PID=2341:0043 SNR=75334323935351D0D022":
       ard_port = p[0]

motor = SabreControl(port=sabre_port)
encoder = EncoderAnalyzer(port=ard_port)
image = ImageAnalyzer(0,show_image=True)
cart = CartPole(motor, encoder, image, encoder)

cart.zeroAngleAnalyzer()
#encoder.setAngleZero()
cart.zeroPosAnalyzer()
cart.goTo(.5)


command_speed = 0
last_time = time.time()
while True:
	observation = cart.getState()
	x,x_dot,theta,theta_dot = observation
	#if np.cos(theta) > -0.75:
	#	cart.goTo(500)
	#	cart.setSpeedMmPerS(0)
	#	continue
	print('obs',observation)
	if not 50 < x < 850:
		cart.goTo(.5)
		cart.setSpeedMmPerS(0)
		last_time = time.time()
		command_speed = 0
		continue

	a = mpc.update(x/1000, x_dot/1000, theta, theta_dot)
	#ulqr(np.array([(x-400)/1000,x_dot/1000,theta,theta_dot]))
	t = time.time() 
	dt = t - last_time
	last_time = t
	command_speed += 1. * a * dt
	#command_speed -= (x - 500) * dt * 0.001 * 0.1
	#command_speed -= x_dot * dt * 0.001 * 0.5
	cart.setSpeedMmPerS(1000 * command_speed)
	#print("theta {}\ttheta_dot {}\taction {}\tspeed {}".format(theta, theta_dot, a, command_speed))
