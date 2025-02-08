import face_recognition
import pickle
import os
from PIL import Image
import numpy as np
import cv2 

data = {}

print("Training model...")

for student in os.listdir("dataset"):
    student_path = os.path.join("dataset", student)
    images = []

    if not os.path.isdir(student_path):
        print(f"Skipping {student_path} (Not a directory)")
        continue

    for img_file in os.listdir(student_path):
        img_path = os.path.join(student_path, img_file)

        try:
            try:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as e_cv2:
                try:
                    pil_image = Image.open(img_path)
                    img = np.array(pil_image.convert('RGB'))[:, :, ::-1].copy()  # Most robust PIL method
                except Exception as e_pil:
                    print(f"Error processing {img_path} (OpenCV): {e_cv2}")
                    print(f"Error processing {img_path} (PIL): {e_pil}")
                    continue

            encodings = face_recognition.face_encodings(img)

            if encodings:
                images.append(encodings[0])
            else:
                print(f"No faces found in {img_path}, skipping.")

        except Exception as e:
            print(f"General error processing {img_path}: {e}")

    if images:
        data[student] = images
    else:
        print(f"No valid images found for {student}")

with open("face_data.pkl", "wb") as f:
    pickle.dump(data, f)

print("Model training complete!")