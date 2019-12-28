#!/usr/bin/env python3

import time
import cv2
import os
# read the user name to creeate folder and picture
with open("user.txt") as myfile:
    username = str.strip(myfile.readline())

output_dir = f"images/{username}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# open webcam
web_cam = cv2.VideoCapture(0)

cascPath = "Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

count = 0

while(True):
    _, imagen_marco = web_cam.read()

    grises = cv2.cvtColor(imagen_marco, cv2.COLOR_BGR2GRAY)

    face = faceCascade.detectMultiScale(grises, 1.5, 5)
    if len(face) < 1:
        print("No faces found.")
    for(x, y, w, h) in face:
        cv2.rectangle(imagen_marco, (x, y), (x+w, y+h), (255, 0, 0), 4)
        count += 1
        file_out_name = f"{output_dir}/{username}/{username}_{count}.jpg"
        print(f"Outputting to {file_out_name}")
        cv2.imwrite(file_out_name, grises[y:y+h, x:x+w])
        cv2.imshow("Creating User - Press Q to exit", imagen_marco)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif count >= 100:
        break
    # Limit retry rate on reading a new frame to 24 frames per second
    time.sleep(1 / 24.0)

# release
web_cam.release()
cv2.destroyAllWindows()
