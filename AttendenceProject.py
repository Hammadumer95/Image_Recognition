import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# from PIL import ImageGrab
 
path = 'ImageAttendence' #Folder Containing Images
images = []     # LIST CONTAINING ALL THE IMAGES
classNames = []    # LIST CONTAINING ALL THE CORRESPONDING CLASS Names
myList = os.listdir(path)
print("Total Classes Detected:",len(myList))


for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
print(classNames)



def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        #myDataList = f.readable()
        myDataList = f.readlines()
#        print(myDataList)
#markAttendance('a')
        nameList =[]
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
           
        if name not in  nameList:
            now = datetime.now()
            d_string =now.date()            
            dt_string = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dt_string},{d_string}')

encodeListKnown = findEncodings(images)
print('Encodings Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
       matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
       faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
       print(faceDis)
       matchIndex = np.argmin(faceDis)
       
       
       if matches[matchIndex]:
          name = classNames[matchIndex].upper()
          print(name)
          y1,x2,y2,x1=faceLoc
          y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
          cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
          cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
          cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
          markAttendance(name)
       
   
        
    cv2.imshow('webcam',img)
    cv2.waitKey(1)
    
#It marks one person attendence one time. we need to mark attendence 2 times a day Check in Check out

#Fake Faces detection is also a problem in this project.. 



