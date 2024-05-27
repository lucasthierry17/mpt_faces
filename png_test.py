import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import gdown
import os
import csv
import time

cap = cv.VideoCapture(0)

ret, frame = cap.read()

plt.imshow(frame)

cap.release()

# Pfad zur XML-Datei relativ zum Skriptverzeichnis
xml_file_path = "haarcascade_frontalface_default.xml"

# Überprüfen, ob die XML-Datei bereits vorhanden ist
if not os.path.exists(xml_file_path):
    # Wenn die Datei nicht vorhanden ist, laden Sie sie von Google Drive herunter
    gdown.download(
        "https://drive.google.com/uc?id=1N5j5ke98qCt_0J70wg6F8diHrF5qqxeX&export=download",
        xml_file_path,
        quiet=False,
    )
    print("XML file downloaded successfully.")
else:
    print("XML file already exists.")

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Laden der Gesichtserkennungsklassifikatoren
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

# Create folder to store faces if it doesn't exist
objects_folder = "objects"
if not os.path.exists(objects_folder):
    os.mkdir(objects_folder)
folder = "test"
person_folder = os.path.join(objects_folder, folder)
if not os.path.exists(person_folder):
    os.mkdir(person_folder)

# Variables for saving mechanism
frame_count = 0
save_frame = True
save_interval = 10  # Speichern alle 10 Sekunden

# Zeitpunkt des letzten Speichervorgangs
last_save_time = time.time()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangle around faces and save images periodically
    for x, y, w, h in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if save_frame:
            cv.imwrite(os.path.join(person_folder, f"frame_{frame_count}.png"), frame)
            with open(
                os.path.join(person_folder, f"frame_{frame_count}.csv"), "w", newline=""
            ) as csvfile:
                writer = csv.writer(csvfile, delimiter=",")
                writer.writerow([x, y, w, h])
            frame_count += 1
            if frame_count >= 30:
                save_frame = False

    # Show frame with faces
    cv.imshow("frame", frame)

    # Prüfen, ob es Zeit ist, einen Speichervorgang durchzuführen
    current_time = time.time()
    if current_time - last_save_time >= save_interval:
        save_frame = True
        last_save_time = current_time

    # Check for 'q' key to quit
    if cv.waitKey(1) == ord("q"):
        break

# Release video capture and close all windows
cap.release()
cv.destroyAllWindows()
