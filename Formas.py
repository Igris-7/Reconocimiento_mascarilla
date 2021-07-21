#deteccion de figuras geometricas
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 10, 150)
    canny = cv2.dilate(canny, None, iterations = 1)
    canny = cv2.erode(canny, None, iterations= 1)

    cnts,_ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        epsilon = 0.01*cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        #print (len(approx))
        x,y,w,h = cv2.boundingRect(approx)

        if len(approx) == 3:
            cv2.putText(frame, 'LOGO', (x,y-5),1,1,(0,255,0),2)
            cv2.drawContours(frame, [approx], 0, (0,255,0),3)

            
    cv2.imshow('Logo',frame)

    k = cv2.waitKey(30) & 0xff
    if k ==27:
        break

cap.release()
cv2.destroyAllWindows()
