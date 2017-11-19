from cart_command import CartCommand
from time import sleep

cart = CartCommand()


cart.toggleEnable()

while True:
	cart.move(-128, 2000)
	sleep(.03)
	cart.move(128, 2000)
	sleep(.03)