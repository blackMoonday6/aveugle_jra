import numpy as np
import cv2
import math


cap = cv2.VideoCapture(0)
while(True):
    lista=[]
    
    linefinale=[]
    ret, frame = cap.read()
    
    imc=frame[340:480,0:640]
    edges=cv2.Canny(imc,100,150)

    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,maxLineGap=200)
    lin=0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            longeur=math.sqrt(((x1-x2)**2)+((y1-y2)**2))
            angle=math.atan(((y2-y1)/(x2-x1)))
            p=((x1+((x2-x1)/2)),(y1+((y2-y1)/2)))
            if p[0]<200:
                k=0
            else:
                k=1
            lista.append((longeur,angle,p,lin,k))
            lin+=1
    lista=sorted(lista, reverse=True,key=lambda x: x[0])
    if len(linefinale)<1:
        
        linefinale.append(lista[0])
    for element in range(len(lista)):
        if linefinale[0][4]!=lista[element][4]:
            linefinale.append(lista[element])
            break

    if len(linefinale)>1:
        x1d=lines[linefinale[0][3]][0][0]
        x2d=lines[linefinale[0][3]][0][1]
        y1d=lines[linefinale[0][3]][0][2]
        y2d=lines[linefinale[0][3]][0][3]
        x1g=lines[linefinale[1][3]][0][0]
        x2g=lines[linefinale[1][3]][0][1]
        y1g=lines[linefinale[1][3]][0][2]
        y2g=lines[linefinale[1][3]][0][3]
        anglefinale=(linefinale[0][1]+linefinale[1][1])/2
        cv2.line(imc,(x1d,x2d),(y1d,y2d),(0,255,0),5)
        cv2.line(imc,(x1g,x2g),(y1g,y2g),(0,255,0),5)
        anglefinale=int((anglefinale/3.14)*180)
        print(anglefinale)
    
    cv2.imshow('frame',frame)
    cv2.imshow('edges',edges)
    cv2.imshow('imc',imc)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#cv2.imwrite('kangg.jpg',frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()