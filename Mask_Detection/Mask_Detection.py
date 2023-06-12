import time

import cv2
import numpy as np
from  cvzone import ClassificationModule as cm
side_face=cv2.CascadeClassifier('face_detected/haarcascade_profileface.xml')
front_face=cv2.CascadeClassifier('face_detected/haarcascade_frontalface_alt.xml')
video=cv2.VideoCapture(0)
image=np.ones((300,300,3),dtype=np.uint8)*255
count = 0
classify=cm.Classifier('model/keras_model.h5','model/labels.txt')
label=['Mask Detected','Wear Mask']
while True:
    s,im=video.read()
    im1=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    front_face_detect=front_face.detectMultiScale(im1,1.35,1)
    side_face_detect=side_face.detectMultiScale(im1,1.35,1)
    X=[]
    Y=[]
    H=[]
    W=[]
    if(len(front_face_detect)!=0):
        for x, y, h, w in front_face_detect:
            X.append(x)
            Y.append(y)
            H.append(h)
            W.append(w)

            # cv2.rectangle(im,(x-30,y-30),(x+h+30,y+w+30),(0,0,255),2)
    else:
        for x, y, h, w in side_face_detect:
            X.append(x)
            Y.append(y)
            H.append(h)
            W.append(w)
            # cv2.rectangle(im, (x - 30, y - 30), (x + h + 30, y + w + 30), (0, 0, 255), 2)
    if(len(X)!=0 and len(Y)!=0 and len(H)!=0 and len(W)!=0):
        new = im[Y[0]:H[0] + Y[0], X[0]:X[0] + W[0]]
        new1=cv2.resize(new,(250,300))
        shape = new1.shape
        image[:shape[0],:shape[1]]=new1
        prediction, index = classify.getPrediction(im,draw=False)
        if(cv2.waitKey(1)==ord('s')):

            count=count+1
            print(count)
            cv2.imwrite(f'Without_Mask/mask_{time.time()}.jpg',image)

        cv2.imshow('white',image)
        cv2.putText(im,label[index],(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0))
    cv2.imshow('Video', im)
    if (cv2.waitKey(1) & 0xFF == ord('z')):
        break;


