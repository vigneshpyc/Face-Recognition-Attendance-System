import face_recognition
import cv2
import pandas as pd
import pickle
import datetime

# Load trained face encodings
with open("face_data.pkl", "rb") as f:
    data = pickle.load(f)

# Open Camera
cam = cv2.VideoCapture(0)

# Load existing attendance or create new
attendance_file = "attendance.csv"
try:
    attendance_df = pd.read_csv(attendance_file)
except FileNotFoundError:
    attendance_df = pd.DataFrame(columns=["Name", "Time"])

print("Starting attendance system...")

while True:
    ret, frame = cam.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for encoding in face_encodings:
        for student, encodings in data.items():
            matches = face_recognition.compare_faces(encodings, encoding, tolerance=0.5)
            if True in matches:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Mark attendance if not already marked today
                if not ((attendance_df["Name"] == student) & (attendance_df["Time"].str.startswith(timestamp[:10]))).any():
                    new_entry = {"Name": student, "Time": timestamp}
                    attendance_df = attendance_df._append(new_entry, ignore_index=True)
                    attendance_df.to_csv(attendance_file, index=False)

                cv2.putText(frame, student, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) == 27:  # Press ESC to exit
        break
cam.release()
cv2.destroyAllWindows()
print("Attendance marked successfully!")