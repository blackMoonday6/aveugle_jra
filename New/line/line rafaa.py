import numpy as np
import cv2

cap = cv2.VideoCapture(0)
while(True):
   
    ret, frame = cap.read()
    
    imc=frame[271:480,10:640]
    edges=cv2.Canny(imc,75,150)

    lines = cv2.HoughLinesP(edges,1,np.pi/180,50,maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(imc,(x1, y1), (x2, y2),(0,255,0),5)
    cv2.imshow('frame',frame)
    cv2.imshow('imc',edges)
    cv2.imshow('imc',imc)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#cv2.imwrite('kangg.jpg',frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()