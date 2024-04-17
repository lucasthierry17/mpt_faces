import numpy as np
import cv2 as cv
import os
import gdown
import uuid
import csv
from common import ROOT_FOLDER
#from cascade import create_cascade

# Quellen
#  - How to open the webcam: https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
#  - How to run the detector: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html
#  - How to download files from google drive: https://github.com/wkentaro/gdown
#  - How to save an image with OpenCV: https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html
#  - How to read/write CSV files: https://docs.python.org/3/library/csv.html
#  - How to create new folders: https://www.geeksforgeeks.org/python-os-mkdir-method/

# This is the data recording pipeline
#def record(args):
    # : Implement the recording stage of your pipeline
    #   Create missing folders before you store data in them (os.mkdir)
    #   Open The OpenCV VideoCapture Device to retrieve live images from your webcam (cv.VideoCapture)
    #   Initialize the Haar feature cascade for face recognition from OpenCV (cv.CascadeClassifier)
    #   If the cascade file (haarcascade_frontalface_default.xml) is missing, download it from google drive
    #   Run the cascade on every image to detect possible faces (CascadeClassifier::detectMultiScale)
    #   If there is exactly one face, write the image and the face position to disk in two seperate files (cv.imwrite, csv.writer)
    #   If you have just saved, block saving for 30 consecutive frames to make sure you get good variance of images.
    #if args.folder is None:
    #    print("Please specify folder for data to be recorded into")
    #    exit()


def record(folder):
    # Pfad zur XML-Datei relativ zum Skriptverzeichnis
    xml_file_path = 'haarcascade_frontalface_default.xml'

    # Überprüfen, ob die XML-Datei bereits vorhanden ist
    if not os.path.exists(xml_file_path):
        # Wenn die Datei nicht vorhanden ist, laden Sie sie von Google Drive herunter
        gdown.download('https://drive.google.com/uc?id=1N5j5ke98qCt_0J70wg6F8diHrF5qqxeX&export=download', xml_file_path, quiet=False)
        print("XML file downloaded successfully.")
    else:
        print("XML file already exists.")

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Laden der Gesichtserkennungsklassifikatoren
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Create folder to store faces if it doesn't exist
    objects_folder = 'objects'
    if not os.path.exists(objects_folder):
        os.mkdir(objects_folder)

    #folder = "test"
    person_folder = os.path.join(objects_folder, folder)
    if not os.path.exists(person_folder):
        os.mkdir(person_folder)

    # Variables for saving mechanism
    frame_count = 0
    save_frame = False  # Change to False initially to skip first save

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

        # Draw rectangle around faces and save images
        for (x, y, w, h) in faces:
            padding = 0.2
            new_w = int(w * (1 + padding))
            new_h = int(h * (1 + padding))
            new_x = max(0, x - int((new_w - w) / 2))
            new_y = max(0, y - int((new_h - h) / 2))
            raw_frame = frame.copy()
            cv.rectangle(frame, (new_x, new_y), (new_x + new_w, new_y + new_h), (255, 0, 0), 2)

            if save_frame:
                # Generate UUID for the current frame
                frame_uuid = uuid.uuid4()

                # Save image with UUID filename
                image_filename = f"frame_{frame_uuid}.png"
                cv.imwrite(os.path.join(person_folder, image_filename), raw_frame)

                # Save face position to CSV file with UUID filename
                csv_filename = f"frame_{frame_uuid}.csv"
                with open(os.path.join(person_folder, csv_filename), "w", newline="") as csvfile:
                    writer = csv.writer(csvfile, delimiter=",")
                    writer.writerow([x, y, w, h])

                save_frame = False  # Reset save_frame after saving

        frame_count += 1
        if frame_count >= 30:
            frame_count = 0
            save_frame = True  # Set save_frame to True after 30 frames

        # Show frame with faces
        cv.imshow('frame', frame)

        # Check for 'q' key to quit
        if cv.waitKey(1) == ord('q'):
            break

    # Release video capture and close all windows
    cap.release()
    cv.destroyAllWindows()



