import cv2
import numpy as np
import pickle
from sabretooth_command import CartCommand
from image_analyzer import ImageAnalyzer
from time import time
import matplotlib.pyplot as plt
from threading import Thread

import subprocess as sp
command = [ FFMPEG_BIN,
            '-i', 'myHolidays.mp4',
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vcodec', 'rawvideo', '-']
pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=10**8)
"v4l2src device=/dev/video0 ! ffmpegcolorspace ! video/x-raw-bgr ! fdsink"

# read 420*360*3 bytes (= 1 frame)
raw_image = pipe.stdout.read(420*360*3)
# transform the byte read into a numpy array
image =  numpy.fromstring(raw_image, dtype='uint8')
image = image.reshape((360,420,3))
# throw away the data in the pipe's buffer.
pipe.stdout.flush()

cap = cv2.VideoCapture( "v4l2src device=/dev/video0 ! ffmpegcolorspace ! video/x-raw-bgr  ! appsink")
_, frame = cap.read()
print(frame)
cv2.imshow('image',frame)

analyzer = ImageAnalyzer()

analyzer.cap.release()
analyzer.cap = cv2.VideoCapture( "v4l2src device=/dev/video0 ! ffmpegcolorspace ! video/x-raw-rgb ! appsink")
_, frame = analyzer.cap.read()
cv2.imshow('image',frame)
	#"autovideosrc ! appsink")
cart = CartCommand()

xs = []
cart.toggleEnable()

N = 60

def grabber():
	for i in range(2 * N):
		analyzer.cap.grab()


t1 = Thread(target=grabber, args=())
t2 = Thread(target=grabber, args=())
t3 = Thread(target=grabber, args=())


accs =  5 * np.cos(2 * np.pi * np.arange(N) / 30)

for i in range(N):
	 acc = accs[i] # at 30 fps, this is 1 cycle per second
	 for i in range(3):
	 	analyzer.cap.retrieve()
	 x, dx, theta, dtheta = analyzer.analyzeFrame()
	 xs.append(x)
	 cart.move(acc, 300)
	 key = cv2.waitKey(1)

cv2.destroyAllWindows()
print(xs)
print(accs)
key = cv2.waitKey(1)
correlates = np.correlate(accs, np.array(xs), "full") 
print(np.argmax(np.abs(correlates)))
#print(correlates)

plt.plot(correlates)
plt.show()







