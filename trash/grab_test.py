import cv2
import time
import matplotlib.pyplot as plt
cap = cv2.VideoCapture(1)
cap.set(cv2.cv.CV_CAP_PROP_FPS, 90)
#cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(3)

for i in range(60):
	ret, frame = cap.read()

	cv2.waitKey(1)


t = time.time()
for i in range(60):
	ret, frame = cap.read()
	cv2.imshow('mask top', frame)
	cv2.waitKey(1)
print(60/(time.time() - t))
time.sleep(1)

for i in range(4):
	cap.grab()
t = time.time()
for j in range(60):
	ret, frame = cap.read()
	cv2.imshow('mask top', frame)
	cv2.waitKey(1)
#ret, frame = cap.read()
print(60/(time.time() - t))
'''
time.sleep(1)
ts = []
t = time.time()
for j in range(10):
	time.sleep(1)
	t = time.time()
	for i in range(j):
		cap.grab()
	ts.append(time.time() - t)

plt.plot(ts)
plt.show()
'''