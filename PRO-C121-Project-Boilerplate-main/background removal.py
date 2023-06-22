# import cv2 to capture videofeed
import cv2

import numpy as np

# attach camera indexed as 1
camera = cv2.VideoCapture(1)

# setting framewidth and frameheight as 640 X 480
camera.set(3 , 640)
camera.set(4 , 480)

# loading the mountain image
mountain = cv2.imread('Mount-Everest.jpg')
# resizing the mountain image as 640 X 480
mount_res = cv2.resize(mountain,(640,480))

while True:

    # read a frame from the attached camera
    status , frame = camera.read()

    # if we got the frame successfully
    if status:

        # flip it
        frame = cv2.flip(frame , 1)

        # converting the image to RGB for easy processing
        frame_rgb = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)

        # creating thresholds
        lower_bound = np.array([100,100,100])#works for white background
        upper_bound = np.array([255,255,255])

        # thresholding image
        mask = cv2.inRange(frame_rgb,lower_bound,upper_bound)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3,3), np.uint8))
        # inverting the mask
        mask_inv = cv2.bitwise_not(mask)
        # bitwise and operation to extract foreground / person and background / mountains respectively
        person = cv2.bitwise_and(frame,frame,mask=mask_inv)
        not_person = cv2.bitwise_and(mount_res,mount_res,mask=mask)
        # final image
        frame = cv2.addWeighted(person,1,not_person,1,0)
        # show it
        cv2.imshow('frame' , frame)

        # wait of 1ms before displaying another frame
        code = cv2.waitKey(1)
        if code  ==  32:
            break

# release the camera and close all opened windows
camera.release()
cv2.destroyAllWindows()
