import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier("classifier\haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(1); #0 for laptop webcam, 1 for external webcam;


rec=cv2.createLBPHFaceRecognizer();
rec.load('recognizer\trainningdata.yml')
id=0
font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,1, 0.75, 1.5,1, 2)

#    example of an individual's profile
#--------------------------------------------
#gender_Individual[N] = "Gender: ..."
#occupation_Individual[N] = "Occupation: ... "
#hobbies_Individual[N] = "Hobbies: ..."
#--------------------------------------------

gender_Individual1 = "Gender: Male"
occupation_Individual1 = "Occupation: Student "
hobbies_Individual1 = "Hobbies: Coding"

gender_Individual2 = "Gender: Female"
occupation_Individual2 = "Occupation: Teacher"
hobbies_Individual2 = "Hobbies: Golf"

while True:
    ret,img=cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id, conf= rec.predict(gray[y:y+h,x:x+w])
        
        if(id ==1):
            id = "Individual1"
            cv2.cv.PutText(cv2.cv.fromarray(img),str(id), (x,y+h+30),font, (0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(gender_Individual1), (x,y+h+60),font,(0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(occupation_Individual1),(x,y+h+90),font,(0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(hobbies_Individual1), (x,y+h+120),font,(0,255,0))
            cv2.imshow("Face",img)
        elif(id ==2):
            id = "Individual2"
            cv2.cv.PutText(cv2.cv.fromarray(img),str(id), (x,y+h+30),font, (0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(gender_Individual2), (x,y+h+60),font, (0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(occupation_Individual2),(x,y+h+90),font, (0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(hobbies_Individual2), (x,y+h+120),font, (0,255,0))
            cv2.imshow("Face",img)
            
    if (cv2.waitKey(1) == ord('q')):
        break;

cam.release()
cv2.destroyAllWindows()

#Made by Tiago Goncalves
