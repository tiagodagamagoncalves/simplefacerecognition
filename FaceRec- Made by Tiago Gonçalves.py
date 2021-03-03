import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier("C:\Users\Tiago\Desktop\Face Recognizer               -           tiagogclvs\Classifiers\haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(1);


rec=cv2.createLBPHFaceRecognizer();
rec.load('recognizer\\trainningdata.yml')
id=0
font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,1, 0.75, 1.5,1, 2)

genderTiago = "Gender: Male"
occupationTiago = "Occupation: Jogador de Futebol"
hobbiesTiago = "Hobbies: Tocar piano"
criminalTiago = "Criminal Records: 0"

genderAlex = "Gender: Male"
occupationAlex = "Occupation: Student"
hobbiesAlex = "Hobbies: Stealing Delivery Companies"
criminalAlex = "Criminal Records: Beat a camel"

genderMiguel = "Gender: Male"
occupationMiguel = "Occupation: Teacher"
hobbiesMiguel = "Hobbies: Planting trees"
criminalMiguel = "Criminal Records: Killed a student"

genderTomas = "Gender: Male"
occupationTomas = "Occupation: Student"
hobbiesTomas = "Hobbies: Soccer"

genderRic = "Gender: Male"
occupationRic = "Occupation: Student"
hobbiesRic = "Hobbies: Gaming"

genderManel = "Gender: Male"
occupationManel = "Occupation: Student"
hobbiesManel = "Hobbies: Karate"

genderCam = "Gender: Female"
occupationCam = "Occupation: Student"
hobbiesCam = "Hobbies: Music"

genderGC = "Gender: Male"
occupationGC = "Occupation: Student"
hobbiesGC = "Hobbies: Soccer"

genderSa = "Gender: unknown"
occupationSa = "Occupation: Night Worker / Student"
hobbiesSa = "Hobbies: selling things"

while True:
    ret,img=cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id, conf= rec.predict(gray[y:y+h,x:x+w])
        if(id ==1):
            id = "Julio Tomas"
            cv2.cv.PutText(cv2.cv.fromarray(img),str(id), (x,y+h+30),font, (0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(genderTiago), (x,y+h+60),font,(0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(occupationTiago),(x,y+h+90),font,(0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(hobbiesTiago), (x,y+h+120),font,(0,255,0))
            cv2.imshow("Face",img)
        elif(id ==2):
            id = "Alexandre Salgado"
            cv2.cv.PutText(cv2.cv.fromarray(img),str(id), (x,y+h+30),font, (0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(genderAlex), (x,y+h+60),font, (0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(occupationAlex),(x,y+h+90),font, (0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(hobbiesAlex), (x,y+h+120),font, (0,255,0))
            cv2.imshow("Face",img)
        elif(id ==3):
            id = "Miguel Dias"
            cv2.cv.PutText(cv2.cv.fromarray(img),str(id), (x,y+h+30),font, (0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(genderMiguel), (x,y+h+60),font, (0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(occupationMiguel),(x,y+h+90),font,(0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(hobbiesMiguel), (x,y+h+120),font,(0,255,0))
            cv2.imshow("Face",img)
        elif(id ==4):
            id = "Tomas Martins"
            cv2.cv.PutText(cv2.cv.fromarray(img),str(id), (x,y+h+30),font, (0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(genderTomas), (x,y+h+60),font, (0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(occupationTomas),(x,y+h+90),font, (0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(hobbiesTomas), (x,y+h+120),font, (0,255,0))
            cv2.imshow("Face",img)
        elif(id ==5):
            id = "Ricardo Cardoso"
            cv2.cv.PutText(cv2.cv.fromarray(img),str(id), (x,y+h+30),font, (0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(genderRic), (x,y+h+60),font, (0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(occupationRic),(x,y+h+90),font, (0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(hobbiesRic), (x,y+h+120),font, (0,255,0))
            cv2.imshow("Face",img)
        elif(id ==6):
            id = "Manuel Simoes"
            cv2.cv.PutText(cv2.cv.fromarray(img),str(id), (x,y+h+30),font, (0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(genderManel), (x,y+h+60),font, (0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(occupationManel),(x,y+h+90),font, (0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(hobbiesManel), (x,y+h+120),font, (0,255,0))
            cv2.imshow("Face",img)
        elif(id ==7):
            id = "Camila Lourenco"
            cv2.cv.PutText(cv2.cv.fromarray(img),str(id), (x,y+h+30),font, (0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(genderCam), (x,y+h+60),font, (0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(occupationCam),(x,y+h+90),font, (0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(hobbiesCam), (x,y+h+120),font, (0,255,0))
            cv2.imshow("Face",img)
        elif(id ==8):
            id = "Goncalo Costa"
            cv2.cv.PutText(cv2.cv.fromarray(img),str(id), (x,y+h+30),font, (0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(genderGC), (x,y+h+60),font, (0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(occupationGC),(x,y+h+90),font, (0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(hobbiesGC), (x,y+h+120),font, (0,255,0))
            cv2.imshow("Face",img)
        elif(id ==9):
            id = "Sardao da Noite"
            cv2.cv.PutText(cv2.cv.fromarray(img),str(id), (x,y+h+30),font, (0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(genderSa), (x,y+h+60),font, (0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(occupationSa),(x,y+h+90),font, (0,255,0))
            cv2.cv.PutText(cv2.cv.fromarray(img),str(hobbiesSa), (x,y+h+120),font, (0,255,0))
            cv2.imshow("Face",img)
            
    if (cv2.waitKey(1) == ord('q')):
        break;

cam.release()
cv2.destroyAllWindows()

#Made by Tiago Goncalves
