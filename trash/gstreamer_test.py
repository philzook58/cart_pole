import cv2
import numpy as np
import pickle
import time
from threading import Thread

import subprocess as sp
#command = [ FFMPEG_BIN,
#           '-i', 'myHolidays.mp4',
#            '-f', 'image2pipe',
#            '-pix_fmt', 'rgb24',
#            '-vcodec', 'rawvideo', '-']

#command = ["gst-launch-1.0", "-v", "v4l2src device=/dev/video0 ! video/x-raw,framerate=30/1, width=640, height=480 ! fdsink"]
command = "gst-launch-1.0 -v v4l2src device=/dev/video0 ! video/x-raw, format=BGR, width=640, height=480 ! videoconvert ! fdsink fd=1"
#pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=10**9, shell=True)

#time.sleep(0.1)
 #we're offset from the right colors consistently

raw_image=None
image = np.zeros((480,640,3))
def readFrame():
	global image, raw_image
	pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=-1, shell=True)
	pipe.stdout.read(3*570 + 1)
	for i in range(100):
		pipe.stdout.flush()
		raw_image = pipe.stdout.read(480*640*3)
		# transform the byte read into a numpy array
		image =  np.fromstring(raw_image, dtype='uint8')
		image = image.reshape((480,640,3))
		cv2.imshow('image',image)
		key = cv2.waitKey(1)

#t1 = Thread(target=readFrame, args=())


def showImage():
	global raw_image
	while True:
		if raw_image:
			image =  np.fromstring(raw_image, dtype='uint8')
			image = image.reshape((480,640,3))
			cv2.imshow('image',image)
			key = cv2.waitKey(1)
import cProfile
cProfile.run('readFrame()')
#readFrame()
#t1 = Thread(target=showImage, args=())
#while True:
	# read 420*360*3 bytes (= 1 frame)
#	pass
	# throw away the data in the pipe's buffer.
	#



pipe.terminate()