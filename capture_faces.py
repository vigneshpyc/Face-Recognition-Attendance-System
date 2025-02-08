import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


student_name = input("Enter Student Name: ")
student_path = f"dataset/{student_name}"  
os.makedirs(student_path, exist_ok=True)  

cam = cv2.VideoCapture(0)  
count = 0

print("Capturing 50 images. Please keep your face steady...")

while count < 50:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        img_path = f"{student_path}/{count}.jpg"
        cv2.imwrite(img_path, face_img)
        count += 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Capture", frame)
    if cv2.waitKey(1) == 27: 
        break

cam.release()
cv2.destroyAllWindows()
print(f"Face capture complete! Images saved in {student_path}/")