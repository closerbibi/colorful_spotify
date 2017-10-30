import cv2
import numpy as np
import pdb
import time
cap = cv2.VideoCapture(0)

# setup initial location of window
y0,h0,x0,w0 = 250,180,400,250  
track_window = (x0,y0,w0,h0)
#time.sleep(2)
'''
# wait until pressing 's', and take the first frame
while(1):
    i = cv2.waitKey(5) & 0xFF
    if i == 115: # 's' ascii
        break
'''
_, frame = cap.read()
block = frame[y0:y0+h0, x0:x0+w0]

# Convert BGR to HSV
hsv_block = cv2.cvtColor(block, cv2.COLOR_BGR2HSV)

# define range of blue color in HSV
lb = np.array([110,100,100])
ub = np.array([130,255,255])

lr1 = np.array([170,150,100])
ur1 = np.array([179,255,255])
lr2 = np.array([0,150,100])
ur2 = np.array([10,255,255])

lg = np.array([50,50,50])
ug = np.array([70,255,255])

# Threshold the HSV image to get only desired colors
maskb = cv2.inRange(hsv_block, lb, ub)
maskr1 = cv2.inRange(hsv_block, lr1, ur1)
maskr2 = cv2.inRange(hsv_block, lr2, ur2)
mask_or = maskb | maskr1
blue_hist = cv2.calcHist([hsv_block],[0],maskb,[180],[0,180])
red_hist = cv2.calcHist([hsv_block],[0],maskr1,[180],[0,180])
hist = cv2.calcHist([hsv_block],[0],mask_or,[180],[0,180])
cv2.normalize(blue_hist,blue_hist,0,255,cv2.NORM_MINMAX)
cv2.normalize(red_hist,red_hist,0,255,cv2.NORM_MINMAX)
cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)
cv2.imshow('mask_or',mask_or)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):

    # Take each frame and counting distance 
    ret , frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv],[0],hist,[0,180],1)
    '''
    if maskb.sum() > maskr1.sum():
        dst = cv2.calcBackProject([hsv],[0],blue_hist,[0,180],1)
    else:
        dst = cv2.calcBackProject([hsv],[0],red_hist,[0,180],1)
    '''
    #pdb.set_trace()
    
    
    # apply meanshift to get the new location
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)

    # Draw it on image
    x,y,w,h = track_window
    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)
    cv2.imshow('img2',frame)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask = maskb | maskr1)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
    cv2.imshow('res',res)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27: # esc ascii
        break

cap.release()
cv2.destroyAllWindows()