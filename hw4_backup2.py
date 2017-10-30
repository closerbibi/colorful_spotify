import cv2
import numpy as np
import pdb
import time
import subprocess

cap = cv2.VideoCapture(0)

# setup initial location of window
y0,h0,x0,w0 = 250,90,550,125  
track_window = (x0,y0,w0,h0)
#time.sleep(1)

# define range of blue color in HSV
lb = np.array([110,100,100])
ub = np.array([130,255,255])

lr1 = np.array([170,150,200])
ur1 = np.array([179,255,255])
#lr2 = np.array([0,150,100])
#ur2 = np.array([10,255,255])
#lg = np.array([50,50,50])
#ug = np.array([70,255,255])

# variable for auxiliary function
idle_count = 0

def tracking_color(hsv,roi_hist,track_window,mask,color):
    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
    # apply meanshift to get the new location
    _, track_window = cv2.meanShift(dst, track_window, term_crit)
        # Draw it on image
    x,y,w,h = track_window
    cv2.rectangle(frame, (x,y), (x+w,y+h), color,2)
    cv2.imshow('frame',frame)

    #show rectangle of mask and UI
    #res = cv2.bitwise_and(frame[y:y+h, x:x+w],frame[y:y+h, x:x+w], mask = mask)
    #cv2.imshow('res',res)
    cv2.circle(img, (x,y), 5, color, -1)
    cv2.imshow('Desktop',img)    
    return track_window
'''
def color_mask():
    _, f_define = cap.read()
    cv2.rectangle(f_define,(x0,y0),(x0+w0,y0+h0),(255,0,0),3)
    cv2.imshow('catch color blue',f_define)
    i = cv2.waitKey(20) & 0xFF
    if i == 98: # 'b' ascii
    #if cv2.waitKey(20) == ord('b'):
        roi_b = f_define[y0:y0+h0, x0:x0+w0]
        # Convert BGR to HSV
        hsv_roi_b = cv2.cvtColor(roi_b, cv2.COLOR_BGR2HSV)
        break
'''
# define color mask
while(1):
    _, f_define = cap.read()
    cv2.rectangle(f_define,(x0,y0),(x0+w0,y0+h0),(255,0,0),3)
    cv2.imshow('catch color blue',f_define)
    i = cv2.waitKey(20) & 0xFF
    if i == 98: # 'b' ascii
    #if cv2.waitKey(20) == ord('b'):
        roi_b = f_define[y0:y0+h0, x0:x0+w0]
        # Convert BGR to HSV
        hsv_roi_b = cv2.cvtColor(roi_b, cv2.COLOR_BGR2HSV)
        break

while(1):
    _, f_define = cap.read()
    cv2.rectangle(f_define,(x0,y0),(x0+w0,y0+h0),(0,0,255),3)
    cv2.imshow('catch color red',f_define)
    i = cv2.waitKey(20) & 0xFF
    if i == 114: # 'r' ascii
        roi_r = f_define[y0:y0+h0, x0:x0+w0]
        # Convert BGR to HSV
        hsv_roi_r = cv2.cvtColor(roi_r, cv2.COLOR_BGR2HSV)
        break

cv2.destroyAllWindows()

# Threshold the HSV image to get only desired colors
roi_maskb = cv2.inRange(hsv_roi_b, lb, ub)
roi_maskr1 = cv2.inRange(hsv_roi_r, lr1, ur1)
#mask_or = maskb | maskr1
roi_hist_b = cv2.calcHist([hsv_roi_b],[0],roi_maskb,[180],[0,180])
roi_hist_r = cv2.calcHist([hsv_roi_r],[0],roi_maskr1,[180],[0,180])
cv2.normalize(roi_hist_b,roi_hist_b,0,255,cv2.NORM_MINMAX)
cv2.normalize(roi_hist_r,roi_hist_r,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
'''
# showing interface
while(1):
    img = cv2.imread('windows.jpg',1)
    cv2.imshow('press 's' to start ',img)
    i = cv2.waitKey(20) & 0xFF
    if i == 115: # 's' ascii
        break
'''
while(1):
    # showing interface
    img = cv2.imread('windows.jpg',1)

    # Take each frame and counting distance 
    _, frame = cap.read()

    # hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    x,y,w,h = track_window
    maskb = cv2.inRange(hsv[y:y+h, x:x+w], lb, ub)
    maskr1 = cv2.inRange(hsv[y:y+h, x:x+w], lr1, ur1)
    #pdb.set_trace()    

    sum_b = np.sum(np.sum(maskb))
    sum_r = np.sum(np.sum(maskr1))
    # blue case
    if sum_b <= sum_r:
        color =  0,0,255
        track_window = tracking_color(hsv,roi_hist_b,track_window,maskb,color)
        idle_count = 0

    # red case
    elif sum_b > sum_r:
        color =  255,0,0
        track_window = tracking_color(hsv,roi_hist_r,track_window,maskr1,color)
        idle_count = idle_count +1

    if idle_count >= 50
        subprocess.call("command-name-here")

    #pdb.set_trace()

    k = cv2.waitKey(5) & 0xFF
    if k == 27: # esc ascii
        break

cap.release()
cv2.destroyAllWindows()