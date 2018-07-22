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
encoder.setAngleZero()
cart.zeroPosAnalyzer()
cart.goTo(.5)
while True:
	#print('image' , "%02.2f"%cart.angle_analyzer.getAngle(), 'encoder', "%02.2f"%cart.pos_analyzer.getAngle())
	#print('image' , "%02.2f"%cart.angle_analyzer.getAngle(), 
	#	"%02.2f"%cart.angle_analyzer.getAngleVel(),
	#	'encoder' , "%02.2f"%cart.pos_analyzer.getAngle(), 
	#	 "%02.2f"%cart.pos_analyzer.getAngleVel())
	print(cart.getState())

