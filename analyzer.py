import serial
import cv2
import time
import subprocess
import numpy as np
from diffval import DiffVal


class Analyzer():
	pos_range_mm = 880
	motor_mm_s = 2046. / 1000. # motor moves roughly 1m/s at full tilt?
	def __init__(self):
		self.pos = DiffVal(0)
		self.angle = DiffVal(0)

		self.time = DiffVal(0)

		self.pos_zero = 0
		self.pos_max = 1

		self.angle_zero = 0



	def setPosZero(self):
		self.pos_zero = self.pos.val

	def setPosMax(self):
		self.pos_max = self.pos.val

	def setAngleZero(self):
		self.angle_zero = self.angle.val

	def getPos(self, update=True):
		if update:
			self.updateState()
		pos_range = self.pos_max - self.pos_zero
		return (self.pos.val - self.pos_zero) / pos_range

	def getPosMm(self, update=True):
		return self.getPos(update) * self.pos_range_mm

	def getVel(self, update=True):
		if update:
			self.updateState()
			time.sleep(1./500)
			self.updateState()
		pos_range = self.pos_max - self.pos_zero
		return (self.pos.delta()) / (pos_range * self.time.delta())

	def getVelMm(self, update=True):
		return self.getVel(update) * self.pos_range_mm


	def getAngle(self, update=True):
		if update:
			self.updateState()
		return (self.angle.val - self.angle_zero)

	def getAngleVel(self, update=True):
		if update:
			self.updateState()
			time.sleep(1./500)
			self.updateState()
		return (self.angle.delta())/ self.time.delta()

	def updateState(self):
		self.time.val = time.time()

class ImageAnalyzer(Analyzer):
	def __init__(self, dev, show_image=False):
		super().__init__()
		print(cv2.__version__)
		self.cap = cv2.VideoCapture(dev)
		print(self.cap.isOpened())
		print(self.cap)

		command = "v4l2-ctl -d %d -c white_balance_automatic=0 -c auto_exposure=1 -c gain_automatic=0"%dev
		output = subprocess.call(command, shell=True)
		command = "v4l2-ctl -d %d -c exposure=10 -c gain=0"%dev
		output = subprocess.call(command, shell=True)
		
		self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1.)
		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320.)
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240.)
		self.cap.set(cv2.CAP_PROP_FPS, 187.)
		
		self.show_image = show_image

		self.last_theta = 0
		self.updateState()
		self.updateState()

	def __del__(self):
		cv2.destroyAllWindows()
		self.cap.release()


	def updateState(self):
		super().updateState()
		ret, frame = self.cap.read()
		assert frame is not None, "frame is none"
		imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)

		contour = max(contours, key=cv2.contourArea)

		rect = cv2.minAreaRect(contour)
		box = cv2.boxPoints(rect)
		box = np.int0(box)

		(x,y), (w, h), theta = rect
		theta = -theta * np.pi / 180

		if w > h:
			theta = theta + np.pi / 2.

		self.angle.val += ((theta - self.angle.val + np.pi / 2.) % np.pi) - np.pi / 2.

		#cv2.drawContours(frame,contours,-1,(0,0,255),2)
		cv2.drawContours(frame,[box],0,(0,0,255),2)
		if self.show_image:
			cv2.imshow('frame', frame)
		cv2.waitKey(1)



class EncoderAnalyzer(Analyzer):
	rad_per_pulse = 2 * 3.1459 / 1200.
	def __init__(self, port="/dev/ttyACM0"):
		super().__init__()
		self.ser = serial.Serial(port, 115200, timeout=1, parity=serial.PARITY_EVEN, stopbits=serial.STOPBITS_ONE)  
		print("Initialized Analyzer Serial")

		self.end_stop_low = False
		self.end_stop_high = False		


	def getEndStopHigh(self, update=True):
		if update:
			self.updateState()
		return self.end_stop_high

	def getEndStopLow(self, update=True):
		if update:
			self.updateState()
		return self.end_stop_low

	def updateState(self):
		super().updateState()
		self.ser.write(b'a')
		line = self.ser.readline()
		try:
			lines = line.split(b'\t')
			if len(lines) == 4:
				self.pos.val = int(lines[0])

				self.angle.val = int(lines[1]) * self.rad_per_pulse

				self.end_stop_high = (int(lines[3]) > 0)
				self.end_stop_low = (int(lines[2]) > 0) 
			else:
				print("Length of arduino message not 4")
		except ValueError:
			print("Bad Serial Response")