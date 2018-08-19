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

gravity = 9.81
mass_pole = 0.15
length = 0.5
moment_of_inertia = (1./3.) * mass_pole * length**2

def E(x): # energy
	return (moment_of_inertia * x[3]**2 / 2) - (np.cos(x[2]) * length * mass_pole * gravity / 2)


Ed = E([0,0,np.pi,0])

def u(x):
	return  1.0 * (E(x)-Ed-0.3) * x[3] * np.cos(x[2])

A = np.array([
    [0,1,0,0],
    [0,0,0,0],
    [0,0,0,1],
    [0,0,length * mass_pole * gravity / (2 * moment_of_inertia) ,0]
	])
B = np.array([0,1,0,length * mass_pole / (2 * moment_of_inertia)]).reshape((4,1))
Q = np.diag([4.0, 1.0, 1.0, 0.01])
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
	x,x_dot,theta,theta_dot = cart.getState()
	observation = np.array([(x-400)/1000.,x_dot/1000.,theta,theta_dot])
	print('obs', observation)
	if not 100 < x < 800:
		cart.goTo(.5)
		cart.setSpeedMmPerS(0)
		command_speed = 0
		last_time = time.time()

		#break

	if  abs(E(observation)-Ed) < 0.05 and np.cos(observation[2]) < -0.9 and abs(command_speed)<4: # balance
		print("linear control")
		a = 1.0 * ulqr(observation)[0]
	else:
		a = 0.5*(u(observation)/0.15 - 10.0 * observation[0] -  1.0 * observation[1]) # swing up

	t = time.time() 
	dt = t - last_time
	last_time = t
	command_speed += 1 * a * dt

	cart.setSpeedMmPerS(1000 *command_speed)
	p#rint("theta %06.2f\ttheta_dot %06.2f\taction %06.2f\tspeed %06.2f" % (theta, theta_dot, a, command_speed))
