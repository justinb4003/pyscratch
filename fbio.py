import cv2
import os
#read the user name to creeate folder and picture
myfile = open("user.txt", "rt") 
contents = myfile.read()         
myfile.close()                   
#variables
var_name = contents
var_folder = contents
#Create root folder for specific user from variable
os.makedirs("images/"+var_name)

#open webcam
web_cam = cv2.VideoCapture(0)

cascPath = "Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

count = 0

while(True):
    _, imagen_marco = web_cam.read()

    grises = cv2.cvtColor(imagen_marco, cv2.COLOR_BGR2GRAY)

    face = faceCascade.detectMultiScale(grises, 1.5, 5)

    for(x,y,w,h) in rostro:
        cv2.rectangle(imagen_marco, (x,y), (x+w, y+h), (255,0,0), 4)
        count += 1
        path = "images"
        folder = 'var_folder'
        name = 'var_name'
        cv2.imwrite(path+folder+name+str(count)+".jpg", grises[y:y+h, x:x+w])
        cv2.imshow("Creating User - Press Q to exit", imagen_marco)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    elif count >= 100:
        break


# release
web_cam.release()
cv2.destroyAllWindows()
