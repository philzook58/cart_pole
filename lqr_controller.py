import time
import numpy as np
import serial.tools.list_ports
import scipy.linalg as linalg
lqr = linalg.solve_continuous_are



from analyzer import EncoderAnalyzer, ImageAnalyzer
from motor_control import SabreControl
from cart_pole import CartPole
import serial.tools.list_ports

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

gravity = 9.8
mass_pole = 0.1
length = 0.5

moment_of_inertia = (1./3.) * mass_pole * length**2
print(moment_of_inertia)

A = np.array([
    [0,1,0,0],
    [0,0,0,0],
    [0,0,0,1],
    [0,0,length * mass_pole * gravity / (2 * moment_of_inertia) ,0]
	])
B = np.array([0,1,0,length * mass_pole / (2 * moment_of_inertia)]).reshape((4,1))
Q = np.diag([1.0, 1.0, 1.0, 0.01])
R = np.array([[0.001]])

P = lqr(A,B,Q,R)
Rinv = np.linalg.inv(R)
K = np.dot(Rinv,np.dot(B.T, P))
print(K)
def ulqr(x):
	x1 = np.copy(x)
	x1[2] = np.sin(x1[2] + np.pi)
	return -np.dot(K, x1)

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
	if not 100 < x < 800:
		cart.goTo(.5)
		cart.setSpeedMmPerS(0)
		last_time = time.time()
		command_speed = 0
		continue

	a = ulqr(np.array([(x-400)/1000,x_dot/1000,theta,theta_dot]))
	t = time.time() 
	dt = t - last_time
	last_time = t
	command_speed += 1. * a[0] * dt
	#command_speed -= (x - 500) * dt * 0.001 * 0.1
	#command_speed -= x_dot * dt * 0.001 * 0.5
	cart.setSpeedMmPerS(1000 * command_speed)
	#print("theta {}\ttheta_dot {}\taction {}\tspeed {}".format(theta, theta_dot, a, command_speed))
