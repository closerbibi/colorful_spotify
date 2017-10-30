import cv2
import numpy as np
import pdb
import time
import subprocess

cap = cv2.VideoCapture(0)

# setup initial location of window
y0,h0,x0,w0 = 250,45,550,60  
track_window = (x0,y0,w0,h0)
#time.sleep(1)

# define range of blue color in HSV
lb = np.array([110,80,80])
ub = np.array([130,255,255])

lr1 = np.array([160,80,80])
ur1 = np.array([179,255,255])
#lr2 = np.array([0,150,100])
#ur2 = np.array([10,255,255])
lg = np.array([50,50,50])
ug = np.array([70,255,255])

# variable for auxiliary function
green_count = 0
blue_count = 0
control_trigger = 0

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
    #res = cv2.bitwise_and(frame,frame, mask = mask)
    cv2.imshow('mask',mask)
    
    #cv2.circle(img, (x,y), 5, color, -1)
    #cv2.imshow('Desktop',img)    
    return track_window

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

while(1):
    _, f_define = cap.read()
    cv2.rectangle(f_define,(x0,y0),(x0+w0,y0+h0),(0,255,0),3)
    cv2.imshow('catch color green',f_define)
    i = cv2.waitKey(20) & 0xFF
    if i == 103: # 'r' ascii
        roi_g = f_define[y0:y0+h0, x0:x0+w0]
        # Convert BGR to HSV
        hsv_roi_g = cv2.cvtColor(roi_g, cv2.COLOR_BGR2HSV)
        break

cv2.destroyAllWindows()

# Threshold the HSV image to get only desired colors
roi_maskb = cv2.inRange(hsv_roi_b, lb, ub)
roi_maskr1 = cv2.inRange(hsv_roi_r, lr1, ur1)
roi_maskg = cv2.inRange(hsv_roi_g, lg, ug)

roi_hist_b = cv2.calcHist([hsv_roi_b],[0],roi_maskb,[180],[0,180])
roi_hist_r = cv2.calcHist([hsv_roi_r],[0],roi_maskr1,[180],[0,180])
roi_hist_g = cv2.calcHist([hsv_roi_g],[0],roi_maskg,[180],[0,180])
cv2.normalize(roi_hist_b,roi_hist_b,0,255,cv2.NORM_MINMAX)
cv2.normalize(roi_hist_r,roi_hist_r,0,255,cv2.NORM_MINMAX)
cv2.normalize(roi_hist_g,roi_hist_g,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 1 )
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
    #img = cv2.imread('windows.jpg',1)

    # Take each frame and counting distance 
    _, frame = cap.read()

    # hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    x1,y1,w1,h1 = track_window
    maskb = cv2.inRange(hsv[y1:y1+h1, x1:x1+w1], lb, ub)
    maskr1 = cv2.inRange(hsv[y1:y1+h1, x1:x1+w1], lr1, ur1)
    maskg = cv2.inRange(hsv[y1:y1+h1, x1:x1+w1], lg, ug)
    #pdb.set_trace()    

    sum_b = np.sum(np.sum(maskb))
    sum_r = np.sum(np.sum(maskr1))
    sum_g = np.sum(np.sum(maskg))

    if sum_g > sum_b and sum_g > sum_r:
        color =  0,255,0
        track_window = tracking_color(hsv,roi_hist_b,track_window,maskb,color)
        green_count = green_count + 1
        control_trigger = 1

    # mean shift, case red
    elif sum_b <= sum_r:
        color =  0,0,255
        track_window = tracking_color(hsv,roi_hist_b,track_window,maskb,color)
        blue_count = 0
        green_count = 0

    # mean shift, red blue
    elif sum_b > sum_r:
        color =  255,0,0
        track_window = tracking_color(hsv,roi_hist_r,track_window,maskr1,color)
        blue_count = blue_count +1

    print blue_count
    if blue_count == 6 and control_trigger != 1:
        subprocess.call(["./shpotify-master/spotify","play"])
        control_trigger = 1
    
    elif blue_count == 6 and control_trigger == 1:
        subprocess.call(["./shpotify-master/spotify","pause"])
        control_trigger = 0
    
    elif green_count == 6:
        song_name = raw_input('Why dont u request some music :')
        subprocess.call(["./shpotify-master/spotify","play",song_name])    
    #pdb.set_trace() 
    
    de_maskb = cv2.inRange(hsv, lb, ub)
    de_maskr1 = cv2.inRange(hsv, lr1, ur1)
    de_mask = de_maskb | de_maskr1
    cv2.rectangle(de_mask, (x1,y1), (x1+w1,y1+h1), (0,0,255),2)
    res1 = cv2.bitwise_and(frame,frame, mask = de_mask)
    cv2.imshow("debug show color mask",res1)
    
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27: # esc ascii
        break
    
cap.release()
cv2.destroyAllWindows()