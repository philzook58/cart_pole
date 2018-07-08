import cv2
import subprocess
import numpy as np

print(cv2.__version__)

cap = cv2.VideoCapture(1)

command = "v4l2-ctl -d 1 -c white_balance_automatic=0 -c auto_exposure=1 -c gain_automatic=0"
output = subprocess.call(command, shell=True)
command = "v4l2-ctl -d 1 -c exposure=0 -c gain=0"
output = subprocess.call(command, shell=True)

print(cap.get(cv2.CAP_PROP_BUFFERSIZE))
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1.)
print(cap.get(cv2.CAP_PROP_BUFFERSIZE))


def getBoxAngle(box):
	partner0 = np.argmin([np.linalg.norm(box[0]-point) for point in box[1:]]) + 1

	



while True:
	ret, frame = cap.read()
	imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


	rect = cv2.minAreaRect(contours[0])
	box = cv2.boxPoints(rect)
	box = np.int0(box)

	print rect[2]

	cv2.drawContours(frame,contours,-1,(0,0,255),2)
	#cv2.drawContours(thresh,[box],0,(0,0,255),2)

	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()