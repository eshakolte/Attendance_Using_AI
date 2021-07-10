import pandas as pd
import numpy as np
import datetime
import time
import cv2
import os
import shutil
import csv
from playsound import playsound
from PIL import Image
from PIL import ImageTk
import tkinter as tk
from tkinter import Message
from tkinter import Text
import tkinter.ttk as ttk
import tkinter.font as font


window = tk.Tk()
window.title("Attendance USING AI")
playsound('welcome.mp3')


path = 'C:\\Users\\esha kolte\\Desktop\\Attendence_Using_AI\\face2.jpg'
img = ImageTk.PhotoImage(Image.open(path))
panel = tk.Label(window, image = img)
panel.pack(side = "bottom", fill = "both", expand = "yes")


message = tk.Label(window, text="Attendance USING AI", bg="white", fg="black", width=21, height=1, font=('times', 30, 'italic bold')) 
message.place(x=10, y=10)

message = tk.Label(window, text=" ", bg="white", fg="black", width=60, height=13, font=('times', 10, 'italic bold')) 
message.place(x=30, y=130)

message = tk.Label(window, text="New Users", bg="white", fg="black", width=25, height=2, font=('times', 20, 'italic bold')) 
message.place(x=40, y=130)

lbl = tk.Label(window, text="Enter ID:", width=13, height=1, fg="black", bg="white", font=('times', 13, ' bold ')) 
lbl.place(x=45, y=195)

txt = tk.Entry(window,width=20, bg="white", fg="black", font=('times', 15, ' bold '))
txt.place(x=183, y=195)

lbl2 = tk.Label(window, text="Enter Name:", width=13, fg="black", bg="white", height=2, font=('times', 13, ' bold ')) 
lbl2.place(x=45, y=226)

txt2 = tk.Entry(window,width=20, bg="white", fg="black", font=('times', 15, ' bold '))
txt2.place(x=183, y=235)

message = tk.Label(window, text=" ", bg="black", fg="black", width=60, height=7 ,font=('times', 10, 'italic bold')) 
message.place(x=30, y=365)

lbl3 = tk.Label(window, text="Attendance : ", width=13, fg="white", bg="black", height=2, font=('times', 13, ' bold ')) 
lbl3.place(x=30, y=395)

message2 = tk.Label(window, text=" ", fg="white", bg="black", activeforeground = "green", width=28, height=5  ,font=('times', 13, ' bold ')) 
message2.place(x=163, y=365)
 
message = tk.Label(window, text="Developed By: ESHA KOLTE" ,bg="white", fg="black", width=26, height=1, font=('times', 30, 'italic bold')) 
message.place(x=200, y=507)


def clear():
    txt.delete(0, 'end')    
    res = ""
    message.configure(text= res)


def clear2():
    txt2.delete(0, 'end')    
    res = ""
    message.configure(text= res)    
    

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False


def TakeImages(): 
    Id=(txt.get())
    name=(txt2.get())
    if(is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                sampleNum=sampleNum+1
                cv2.imwrite("TrainingImage\\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                cv2.imshow('Facial Recognition',img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum>60:
                break
        playsound('Dataset-Created.mp3')
        cam.release()
        cv2.destroyAllWindows() 
        res = "Dataset Created" 
        row = [Id , name]
        with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text= res)
    else:
        if(is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text= res)
        if(name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text= res)


def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Dataset Trained Successfully"
    message.configure(text= res)
    playsound('Dataset-Trained-Successfully.mp3')


def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    faces=[]
    Ids=[]
    for imagePath in imagePaths:
        pilImage=Image.open(imagePath).convert('L')
        imageNp=np.array(pilImage,'uint8')
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids


def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("C:\\Users\\esha kolte\\Desktop\\Attendence_Using_AI\\TrainingImageLabel\\Trainner.yml")
    harcascadePath = "C:\\Users\\esha kolte\\Desktop\\Attendence_Using_AI\\haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("C:\\Users\\esha kolte\\Desktop\\Attendence_Using_AI\\StudentDetails\\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)  
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])    
            if(conf < 50):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
               
            else:
                Id='Unknown'                
                tt=str(Id)  
            if(conf > 75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("C:\\Users\\esha kolte\\Desktop\\Attendence_Using_AI\\ImagesUnknown\\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
        cv2.imshow('Facial Recognition',im) 
        if (cv2.waitKey(1)==ord('q')):
            break
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="C:\\Users\\esha kolte\\Desktop\\Attendence_Using_AI\\Attendance\\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    res = "Attendance Updated"
    message.configure(text= res)
    playsound('Thank-you-Your-attendance-updated.mp3')
    cam.release()
    cv2.destroyAllWindows()
    res=attendance
    message2.configure(text= res)


clearButton = tk.Button(window, text="-", command=clear  ,fg="white"  ,bg="red"  ,width=3  ,height=1 ,activebackground = "Red" ,font=('times', 10, ' bold '))
clearButton.place(x=400, y=195)
clearButton2 = tk.Button(window, text="-", command=clear2  ,fg="white"  ,bg="red"  ,width=3  ,height=1, activebackground = "Red" ,font=('times', 10, ' bold '))
clearButton2.place(x=400, y=235)    
takeImg = tk.Button(window, text="Register", command=TakeImages  ,fg="white"  ,bg="grey"  ,width=10  ,height=1, activebackground = "aqua" ,font=('times', 15, ' bold '))
takeImg.place(x=100, y=280)
trainImg = tk.Button(window, text="Trained", command=TrainImages  ,fg="white"  ,bg="blue"  ,width=10  ,height=1, activebackground = "gold" ,font=('times', 15, ' bold '))
trainImg.place(x=260, y=280)
trackImg = tk.Button(window, text="Mark Attendance", command=TrackImages  ,fg="white"  ,bg="green"  ,width=14  ,height=1, activebackground = "lime" ,font=('times', 15, ' bold '))
trackImg.place(x=20, y=70)

window.mainloop()