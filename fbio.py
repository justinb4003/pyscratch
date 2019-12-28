import cv2
import os
#read the user name to creeate folder and picture
myfile = open("user.txt", "rt") 
contents = myfile.read()         
myfile.close()                   
#variables
# Remove any whitespace if there happens to be some
var_name = str.strip(contents)
var_folder = str.strip(contents)
#Create root folder for specific user from variable
if not os.path.exists("images/"+var_name):
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

    # Change reference to restro to face
    for(x,y,w,h) in face:
        cv2.rectangle(imagen_marco, (x,y), (x+w, y+h), (255,0,0), 4)
        count += 1
        path = "images"
        # These are likely not needed at all, unless you wanted to save them
        # in an actual folder named 'var_folder' and 'var_name' and not
        # the cotents of the variable var_name that was defined up top
        folder = 'var_folder'
        name = 'var_name'
        # String interpolation makes building up the output path much easier
        # to read in my opinion and it makes it very clear to everybody
        # you're buidling a string value
        file_out_name = f"{path}/{var_name}/{var_name}_{count}.jpg"
        print(f"Outputting to {file_out_name}")
        cv2.imwrite(file_out_name, grises[y:y+h, x:x+w])
        cv2.imshow("Creating User - Press Q to exit", imagen_marco)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    elif count >= 100:
        break


# release
web_cam.release()
cv2.destroyAllWindows()
