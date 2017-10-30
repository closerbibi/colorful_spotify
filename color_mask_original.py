import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# define range of blue color in HSV
lb = np.array([100,80,80])
ub = np.array([120,255,255])

lr1 = np.array([160,100,100])
ur1 = np.array([179,255,255])
#lr2 = np.array([0,150,100])
#ur2 = np.array([10,255,255])
#lg = np.array([50,50,50])
#ug = np.array([70,255,255])

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only blue colors
    maskr = cv2.inRange(hsv, lr1, ur1)
    maskb = cv2.inRange(hsv, lb, ub)
    mask = maskb | maskr

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= maskr | maskb)

    #cv2.imshow('frame',frame)
    cv2.imshow('res',res)
    #cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()