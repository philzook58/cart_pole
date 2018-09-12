import time
import numpy as np
import serial.tools.list_ports
import scipy.linalg as linalg


import matplotlib.pyplot as plt



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
#image = ImageAnalyzer(0,show_image=True)
cart = CartPole(motor, encoder, encoder, encoder)

cart.zeroAngleAnalyzer()
#encoder.setAngleZero()
cart.zeroPosAnalyzer()
cart.goTo(.6)

fig = plt.figure()
plt.autoscale(tight=True)
ax = fig.add_subplot(111)
ax.autoscale(enable=True, axis="y", tight=False)

li_x = None

command_speed = 0
last_time = time.time()
while True:
	observation = cart.getState()
	x,x_dot,theta,theta_dot = observation
	print(x, x_dot, theta, theta_dot)
	#if np.cos(theta) > -0.75:
	#	cart.goTo(500)
	#	cart.setSpeedMmPerS(0)
	#	continue
	#print('obs',observation)
	if not 100 < x < 800:
		cart.goTo(.5)
		cart.setSpeedMmPerS(0)
		last_time = time.time()
		command_speed = 0
		continue

	a = mpc.update(x/1000, max(min(x_dot/1000,1),-1), theta, theta_dot) # (theta + np.pi) % (2 * np.pi) - np.pi
	if li_x is None:
		li_x, = ax.plot(mpc.x[mpc.N*0:mpc.N*1])
		li_x_dot, = ax.plot(mpc.x[mpc.N*1:mpc.N*2])
		li_theta, = ax.plot(mpc.x[mpc.N*2:mpc.N*3])
		li_theta_dot, = ax.plot(mpc.x[mpc.N*3:mpc.N*4])
		li_force, = ax.plot(mpc.x[mpc.N*4:mpc.N*5])


		fig.canvas.draw()
		plt.show(block=False)
	else:
		li_x.set_ydata(mpc.x[mpc.N*0:mpc.N*1])		
		li_x_dot.set_ydata(mpc.x[mpc.N*1:mpc.N*2])
		li_theta.set_ydata(mpc.x[mpc.N*2:mpc.N*3])
		li_theta_dot.set_ydata(mpc.x[mpc.N*3:mpc.N*4])
		li_force.set_ydata(mpc.x[mpc.N*4:mpc.N*5])

	ax.relim()
	ax.autoscale_view()

	fig.canvas.draw()

	print("Solver Output: ", a)
	#ulqr(np.array([(x-400)/1000,x_dot/1000,theta,theta_dot]))
	t = time.time() 
	dt = t - last_time
	last_time = t
	try:
		command_speed -= 1.0 * a * dt
		#print("command_speed", command_speed)
		cart.setSpeedMmPerS(1000 * command_speed)
	except TypeError:
		pass

	#command_speed -= (x - 500) * dt * 0.001 * 0.1
	#command_speed -= x_dot * dt * 0.001 * 0.5
	#print("theta {}\ttheta_dot {}\taction {}\tspeed {}".format(theta, theta_dot, a, command_speed))
