import numpy as np
import time
import cv2

def add_HSV_filter(frame,camera):

    # blurring the frame
    blur = cv2.GaussianBlur(frame,(5,5),0)

    # converting RGB to HSV
    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)

    # l_b_r = np.array([60,110,50])
    # u_b_r = np.array([255,255,255])
    # l_b_l=np.array([143,110,50])
    # u_b_l = np.array([255,255,255])

    l_b_r = np.array([65,120,100])
    u_b_r = np.array([255,255,255])
    l_b_l=np.array([65,120,100])
    u_b_l = np.array([255,255,255])

    # l_b=np.array([140,106,0]) # lower limit for blue color
    # u_b= np.array([255,255,255])

    # HSV-filter mask
    # mask=cv2.inRange(hsv,l_b_l,u_b_l)

    if(camera==1):
        mask=cv2.inRange(hsv,l_b_r,u_b_r)
    else:
        mask=cv2.inRange(hsv,l_b_l,u_b_l)

    
    # morphological operation- Opening- Erode followed by Dilate - remove noise
    mask=cv2.erode(mask,None,iterations=2)
    mask=cv2.dilate(mask,None,iterations=2)

    return mask 