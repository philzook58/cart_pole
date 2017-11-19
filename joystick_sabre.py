import pygame
import sys
import time
import socket
import cPickle as pickle
from cart_command import CartCommand

pygame.init()

cart = CartCommand()
cart.toggleEnable()

pygame.joystick.init()
clock = pygame.time.Clock()

print pygame.joystick.get_count()
_joystick = pygame.joystick.Joystick(0)
_joystick.init()
while 1:
	pygame.event.get()

	xdir = _joystick.get_axis(0)

	#rtrigger = _joystick.get_axis(5)
	#ltrigger = _joystick.get_axis(4)
	#print(xdir * 200)

	if abs(xdir) < 0.2:
		xdir = 0.0
	print(xdir * 100)
	cart.setSpeed(xdir * 10)
	#MESSAGE = pickle.dumps([xdir,rtrigger,ltrigger])
	#sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))

	clock.tick(30)
