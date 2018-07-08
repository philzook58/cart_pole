import cv2
import subprocess
import numpy as np
from encoder_analyzer import EncoderAnalyzer
import serial.tools.list_ports
import matplotlib.pyplot as plt
import time


print(cv2.__version__)

cap = cv2.VideoCapture(1)

command = "v4l2-ctl -d 1 -c white_balance_automatic=0 -c auto_exposure=1 -c gain_automatic=0"
output = subprocess.call(command, shell=True)
command = "v4l2-ctl -d 1 -c exposure=0 -c gain=0"
output = subprocess.call(command, shell=True)

print(cap.get(cv2.CAP_PROP_BUFFERSIZE))
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1.)
print(cap.get(cv2.CAP_PROP_BUFFERSIZE))

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320.)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240.)
cap.set(cv2.CAP_PROP_FPS, 187.)

ports = list(serial.tools.list_ports.comports())
print(dir(ports))
for p in ports:
    print(dir(p))
    print(p.device)
    if "Sabertooth" in p.description:
       sabreport = p.device
    else:
       ardPort = p.device

print("Initilizing Analyzer")
analyzer = EncoderAnalyzer(port=ardPort) #"/dev/ttyACM0")

image_angles = []
encoder_angles = []

last_theta = 0

for i in range(10):
	ret, frame = cap.read()
start_time = time.time()
nFrames = 1000
for i in range(nFrames):

	start = time.time()
	observation = analyzer.getState()
	print "time", time.time()-start

	ret, frame = cap.read()
	imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)


	rect = cv2.minAreaRect(contours[0])
	box = cv2.boxPoints(rect)
	box = np.int0(box)

	(x,y), (w, h), theta = rect
	theta = -theta
	if w > h:
		print "adding 90"
		theta = theta + 90.
	if theta-last_theta > 90:
		theta -= 180.
	elif theta-last_theta < -90:
		theta += 180.
	last_theta = theta
	print np.sin(np.pi * theta/180.)
	print("pos", (x,y))
	image_angles.append(np.pi * theta/180.)


	
	x,x_dot,theta,theta_dot = observation
	print theta
	encoder_angles.append(theta)

	#cv2.drawContours(frame,contours,-1,(0,0,255),2)
	cv2.drawContours(thresh,[box],0,(0,0,255),2)

	#cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

end_time=time.time()
print("fps: ", nFrames/(end_time-start_time))


cap.release()
cv2.destroyAllWindows()

plt.plot(image_angles[100:]-np.mean(image_angles[100:]), 'r')
plt.plot(encoder_angles[100:-100]-np.mean(encoder_angles[100:-100]), 'b')
plt.figure("corr")
plt.plot(np.correlate(image_angles[100:]-np.mean(image_angles[100:]), encoder_angles[100:-100]-np.mean(encoder_angles[100:-100]), mode='valid'))
plt.show()