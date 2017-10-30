import cv2
import numpy as np
import pdb
import time
import subprocess

cap = cv2.VideoCapture(0)

# setup initial location of window
y0,h0,x0,w0 = 250,150,550,150  
track_window = (x0,y0,w0,h0)
w_half = int(round(w0/2))
h_half = int(round(h0/2))
#time.sleep(1)

# define range of blue color in HSV
lb = np.array([90,80,80])
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
control_trigger = False
vol_ctrl = False
swift = False
function_ctrl = False

def tracking_color(hsv,roi_hist,track_window,color):
    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
    # apply meanshift to get the new location
    _, track_window = cv2.meanShift(dst, track_window, term_crit)
    # Draw it on image
    x,y,w,h = track_window
    cv2.rectangle(frame, (x,y), (x+w,y+h), color,2)
    cv2.imshow('frame',frame)
    
    #cv2.circle(img, (x,y), 5, color, -1)
    #cv2.imshow('Desktop',img)    
    return track_window

# define color mask: blue
while(1):
    _, f_define = cap.read()
    cv2.rectangle(f_define,(x0,y0),(x0+w0,y0+h0),(255,0,0),3)
    cv2.imshow('catch color blue',f_define)
    i = cv2.waitKey(20) & 0xFF
    if i == 98: # 'b' ascii
    #if cv2.waitKey(20) == ord('b'):
        roi_b = f_define[y0:y0+h0, x0:x0+w0]
        #pdb.set_trace()
        hsv_roi_b = cv2.cvtColor(roi_b, cv2.COLOR_BGR2HSV)
        break
# define green
while(1):
    _, f_define = cap.read()
    cv2.rectangle(f_define,(x0,y0),(x0+w0,y0+h0),(0,255,0),3)
    cv2.imshow('catch color green',f_define)
    i = cv2.waitKey(20) & 0xFF
    if i == 103: # 'g' ascii
        roi_g = f_define[y0:y0+h0, x0:x0+w0]
        #pdb.set_trace()
        hsv_roi_g = cv2.cvtColor(roi_g, cv2.COLOR_BGR2HSV)
        break
# define red
while(1):
    _, f_define = cap.read()
    cv2.rectangle(f_define,(x0,y0),(x0+w0,y0+h0),(0,0,255),3)
    cv2.imshow('catch color red',f_define)
    i = cv2.waitKey(20) & 0xFF
    if i == 114: # 'r' ascii
        roi_r = f_define[y0:y0+h0, x0:x0+w0]
        #pdb.set_trace()
        hsv_roi_r = cv2.cvtColor(roi_r, cv2.COLOR_BGR2HSV)
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
# showing interface
#img = cv2.imread('music.jpg',1)

while(1):

    # Take each frame and counting distance 
    _, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    x1,y1,w1,h1 = track_window
    
    maskb = cv2.inRange(hsv[y1:y1+h1, x1:x1+w1], lb, ub)
    maskr1 = cv2.inRange(hsv[y1:y1+h1, x1:x1+w1], lr1, ur1)
    maskg = cv2.inRange(hsv[y1:y1+h1, x1:x1+w1], lg, ug)
    '''
    cv2.imshow('maskb',maskb)
    cv2.imshow('maskr1',maskr1)
    cv2.imshow('maskg',maskg)
    '''
    sum_b = np.sum(np.sum(maskb))
    sum_r = np.sum(np.sum(maskr1))
    sum_g = np.sum(np.sum(maskg))

    if sum_g > sum_b and sum_g > sum_r:
        color =  0,255,0
        dst = cv2.calcBackProject([hsv],[0],roi_hist_g,[0,180],1)
        # apply meanshift to get the new location
        _, track_window = cv2.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x,y,w,h = track_window
        cv2.rectangle(frame, (x,y), (x+w,y+h), color,2)
        cv2.imshow('frame',frame)
        #track_window = tracking_color(hsv,roi_hist_b,track_window,color)
        green_count = green_count + 1
        control_trigger = False

    # mean shift, case red
    elif sum_b <= sum_r:
        color =  0,0,255
        dst = cv2.calcBackProject([hsv],[0],roi_hist_r,[0,180],1)
        # apply meanshift to get the new location
        _, track_window = cv2.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x,y,w,h = track_window
        cv2.rectangle(frame, (x,y), (x+w,y+h), color,2)
        cv2.imshow('frame',frame)
        #track_window = tracking_color(hsv,roi_hist_b,track_window,color)
        decision = blue_count
        blue_count = 0
        green_count = 0
        if swift == True :
            function_ctrl = True
            swift = False

    # mean shift, red blue
    elif sum_b > sum_r:
        color =  255,0,0
        dst = cv2.calcBackProject([hsv],[0],roi_hist_b,[0,180],1)
        # apply meanshift to get the new location
        _, track_window = cv2.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x,y,w,h = track_window
        cv2.rectangle(frame, (x,y), (x+w,y+h), color,2)
        cv2.imshow('frame',frame)
        blue_count = blue_count +1
        swift = True

    # button function
    if function_ctrl == True and green_count <=3 :
        if decision <= 4 :
            if control_trigger == False :
                subprocess.call(["./shpotify-master/spotify","play"])
                control_trigger = True
        
            elif control_trigger == True :
                subprocess.call(["./shpotify-master/spotify","pause"])
                control_trigger = False

        elif decision >= 10 :
            subprocess.call(["./shpotify-master/spotify","vol","up"])
        
        elif decision < 10 :
            subprocess.call(["./shpotify-master/spotify","vol","down"])

        function_ctrl = False
        
    elif green_count == 3 :
            song_name = raw_input('Why dont u request some music :')
            subprocess.call(["./shpotify-master/spotify","play",song_name]) 
            green_count = green_count + 1 


    #cv2.circle(img, (x+w_half,y+h_half), 5, color, -1)
    #cv2.imshow('Desktop',img)   
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27: # esc ascii
        subprocess.call(["./shpotify-master/spotify","quit"])
        break
    
cap.release()
cv2.destroyAllWindows()