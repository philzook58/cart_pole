import cv2
import subprocess
import numpy as np
import time


print(cv2.__version__)

cap = cv2.VideoCapture(1)

#command = "v4l2-ctl -d 1 -c white_balance_automatic=0 -c auto_exposure=1 -c gain_automatic=0"
#output = subprocess.call(command, shell=True)
#command = "v4l2-ctl -d 1 -c exposure=100 -c gain=10"
#output = subprocess.call(command, shell=True)

print(cap.get(cv2.CAP_PROP_BUFFERSIZE))
cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)
#cap.set(cv2.CAP_PROP_FPS, 187.)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320.)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320.)
cap.set(cv2.CAP_PROP_FPS, 187.)

print(cap.get(cv2.CAP_PROP_BUFFERSIZE))


font = cv2.FONT_HERSHEY_SIMPLEX
start_time = time.time()
Nframes = 1000
for i in range(Nframes):
	ret, frame = cap.read()

	cv2.putText(frame,str(time.time()-start_time),(100,100), font, 1,(255,255,255),2,cv2.LINE_AA)
	cv2.imshow('frame', frame)	 
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
end_time=time.time()
print("fps: ", Nframes/(end_time-start_time))

cv2.imshow('frame', frame)	 
cv2.waitKey(0)