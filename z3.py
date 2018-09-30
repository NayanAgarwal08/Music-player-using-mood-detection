import numpy as np
import cv2
cascade_fn = cv2.CascadeClassifier("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml")
nested_fn  = cv2.CascadeClassifier("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_eye.xml")
facedict = {} 
face_cascade = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
def crop_face(gray, face): #Crop the given face
    for (x, y, w, h) in face:
        faceslice = gray[y:y+h, x:x+w]
        facedict["face%s" %(len(facedict)+1)] = faceslice
    return faceslice

while 1:  
    ret, img = cap.read()
    if ret == True:
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            if len(faces) == 1: #Use simple check if one face is detected, or multiple (measurement error unless multiple persons on image)
                faceslice = crop_face(gray, faces) #slice face from image
                cv2.imshow("detect", faceslice) #display sliced face
            else:
                print("no/multiple faces detected, passing over frame")


        cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    if len(facedict) == 10:
        break

cap.release()
cv2.destroyAllWindows()
