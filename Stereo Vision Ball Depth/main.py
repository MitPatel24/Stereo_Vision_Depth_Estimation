import sys
import cv2
import numpy as np
import time
import imutils
import matplotlib.pyplot as plt

import HSV_filter as hsv
import shape_recognition as shape
import triangulation as tri

cap_right = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap_left=cv2.VideoCapture(1,cv2.CAP_DSHOW)

frame_rate = 120

B=9 # distance between the cameras [cm]
f=6 # camera lense's focal length [mm]
alpha=60 # camera field of view in the horizontal plane [degrees]   (56.6)

count = -1

while(True):
    count += 1
    
    ret_right,frame_right = cap_right.read()
    ret_left,frame_left = cap_left.read()

    # calibration
    # If can't catch any frame, break
    if ret_right==False or ret_left==False:
        # print("not able to catch the frame")
        break

    else: 
        # Applying HSV-Filter
        mask_right = hsv.add_HSV_filter(frame_right,1)
        mask_left = hsv.add_HSV_filter(frame_left,0)

        #Result-frames after applying HSV-filter mask
        res_right = cv2.bitwise_and(frame_right,frame_right,mask=mask_right)
        res_left = cv2.bitwise_and(frame_left,frame_left,mask=mask_left)

        # Applying shape recognition
        circles_right = shape.find_circles(frame_right,mask_right)
        circles_left = shape.find_circles(frame_left,mask_left)

        # Hough transforms can be used aswell or some neural network to do object detection

        ################ calculating ball depth ##############################33
        # If no ball can be caught in one camera show text "TRACKING LOST"
 
        if np.all(circles_right)==None or np.all(circles_left)==None:
            cv2.putText(frame_right,"TRACKING LOST",(70,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            cv2.putText(frame_left,"TRACKING LOST",(70,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)  
        else:
            # Function to calculate depth of object.Outputs vector of all depths in case of several balls.
            # All functions used to find depth is in video presentation
            depth = tri.find_depth(circles_right,circles_left,frame_right,frame_left,B,f,alpha)

            cv2.putText(frame_right,"TRACKING",(70,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(124,254,0),2)  
            cv2.putText(frame_left,"TRACKING",(70,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(124,254,0),2)        
            cv2.putText(frame_right,"Distance: " + str(round(depth,3)),(200,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(124,254,0),2)
            cv2.putText(frame_left,"Distance: " + str(round(depth,3)),(200,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(124,254,0),2)

            # multiply computer value with 205.8 to get real-life depth in [cm]. The factor was found manually.
            print("depth:",depth)

    # show the frames
    frame_right=cv2.resize(frame_right,(500,300))
    frame_left=cv2.resize(frame_left,(500,300))
    mask_right=cv2.resize(mask_right,(500,300))
    mask_left=cv2.resize(mask_left,(500,300))
    cv2.imshow('frame right',frame_right)
    cv2.imshow('frame left',frame_left)
    cv2.imshow('mask right',mask_right)
    cv2.imshow('mask left',mask_left)

    # Hit "q" to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # print("close button pressed")
        break

# Release and destroy all windows before termination
cap_right.release()
cap_left.release()

# cv2.destroyAllWindows()