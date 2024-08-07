import cv2
import numpy as np
#Read XML file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#create video capture object. 0 denotes webcam
cap=cv2.VideoCapture(0)

while True:
    # The input image to be recognised
    _, img=cap.read()

    #Convert the image to gray scale
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #Detect the faces
    faces = face_cascade.detectMultiScale(gray,1.1,4)
                                        #Gray Scale, Image factor, Number of Neighbours
#To construct a rectangle around the face detected
    for (x,y,w,h) in faces:


        # create a mask image of the same shape as input image, filled with 0s (black color)
        mask = np.zeros_like(img)
        # create a white filled ellipse
        mask = cv2.ellipse(mask, (int((x+w)/1.2), y+h),(69,69), 0, 0, -180, (255,255,255),thickness=-1)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)



        # Bitwise AND operation to black out regions outside the mask
        result = np.bitwise_and(img, mask)

        #Converting the final result as HSV inorder to detect colors
        hsv_img = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

        # Draws a rectangle

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Black Color
        low_black = np.array([94, 80, 2])
        high_black = np.array([126, 255, 255])

        MASK = cv2.inRange(hsv_img, low_black, high_black)

        #If the MASK only has black pixels caused due to no black colour in the original image
        if cv2.countNonZero(MASK) == 0:
            print("Beard Not Found")
        else:
            print("Beard Found")



    #Display the Respective Results
    cv2.imshow('Image',img)
    cv2.imshow('Result',result)
    cv2.imshow('MASK', MASK)
    #Wait for key press to close the image
    k=cv2.waitKey(30) &0xff
    #Break when esc key is pressed
    if k==27:
        break

cap.release()