import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path='ImagesAttendance'
images =[]
classnames=[]
mylist=os.listdir(path)
print(mylist)

for cl in mylist:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classnames.append(os.path.splitext(cl)[0])
print(classnames)

def findEncoding(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
def markAttendace(name):
    with open('attendance.csv','r+') as f:
        myDataList=f.readlines()
        namelist=[]
        #print(myDataList)
        for line in myDataList:
            entry=line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now=datetime.now()
            dtSring=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtSring}')








encodeListKnown = findEncoding(images)
print('Encoding Complete')

cap=cv2.VideoCapture(0)

while True:
    success,img=cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame,facesCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name=classnames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1=faceLoc

            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendace(name)
    cv2.imshow('webcam',img)
    cv2.waitKey(1)





