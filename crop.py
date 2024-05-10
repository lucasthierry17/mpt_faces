import cv2 as cv
from common import ROOT_FOLDER, TRAIN_FOLDER, VAL_FOLDER
import os
import csv
import random

# Quellen
#  - How to iterate over all files/folders in one directory: https://www.tutorialspoint.com/python/os_walk.htm
#  - How to add border to an image: https://www.geeksforgeeks.org/python-opencv-cv2-copymakeborder-method/

# This is the cropping of images
#def crop(args):
    # : Crop the full-frame images into individual crops
    #   Create the TRAIN_FOLDER and VAL_FOLDER is they are missing (os.mkdir)
    #   Clean the folders from all previous files if there are any (os.walk)
    #   Iterate over all object folders and for each such folder over all full-frame images 
    #   Read the image (cv.imread) and the respective file with annotations you have saved earlier (e.g. CSV)
    #   Attach the right amount of border to your image (cv.copyMakeBorder)
    #   Crop the face with border added and save it to either the TRAIN_FOLDER or VAL_FOLDER
    #   You can use 
    #
    #       random.uniform(0.0, 1.0) < float(args.split) 
    #
    #   to decide how to split them.
    
# Funktion zum Entfernen aller Dateien in einem Ordner
def clean_folder(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            os.remove(os.path.join(root, file))

# Funktion zum Erstellen von Trainings- und Validierungsordnern
def create_folders():
    for folder in [TRAIN_FOLDER, VAL_FOLDER]:
        if not os.path.exists(folder):
            os.mkdir(folder)

# Funktion zum Ausschneiden der Gesichter
def crop(args):
    # Sicherstellen, dass die Ordner existieren und leer sind
    clean_folder(TRAIN_FOLDER)
    clean_folder(VAL_FOLDER)
    create_folders()

    # Ordner von Personen durchsuchen
    for root, dirs, files in os.walk(ROOT_FOLDER):
        for folder_name in dirs:
            person_folder = os.path.join(root, folder_name)

            # Liste der Bilddateien im aktuellen Ordner erhalten
            image_files = [file for file in os.listdir(person_folder) if file.endswith('.png')]

            # Erstellen von Unterordnern für die Person in den Trainings- und Validierungsordnern
            person_train_folder = os.path.join(TRAIN_FOLDER, folder_name)
            person_val_folder = os.path.join(VAL_FOLDER, folder_name)
            os.makedirs(person_train_folder, exist_ok=True)
            os.makedirs(person_val_folder, exist_ok=True)

            for image_file in image_files:
                # Dateipfad zum aktuellen Bild
                image_path = os.path.join(person_folder, image_file)

                # CSV-Dateipfad zum aktuellen Bild
                csv_file_path = os.path.join(person_folder, f"{os.path.splitext(image_file)[0]}.csv")

                # Sicherstellen, dass die CSV-Datei existiert
                if not os.path.exists(csv_file_path):
                    continue

                # Lesen des Bildes
                frame = cv.imread(image_path)

                # Lesen der CSV-Datei und Extrahieren der Gesichtskoordinaten
                with open(csv_file_path, 'r') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    for row in csv_reader:
                        # Gesichtskoordinaten extrahieren (Annahme: x, y, width, height)
                        x, y, w, h = map(int, row)

                        # Mitte des Bildes 
                        mx = (x + w) / 2
                        my = (y + h) / 2

                        # Größe des Bildes
                        bx = abs(x - w) / 2
                        by = abs(y - h) / 2

                        args.border = float(args.border)

                        # Rand von x und y
                        bwx = int(( 1.0 + args.border) * bx)
                        bwy = int(( 1.0 + args.border) * by)

                        # Bild mit Rand
                        imgwb = cv.copyMakeBorder(frame, bwx, bwx, bwy, bwy, cv.BORDER_REFLECT)

                        # Koordinaten vom Bild
                        x = int(mx - (bwx / 2))
                        y = int(my - (bwy / 2))
                        w = int(mx + (bwx * 2))
                        h = int(my + (bwy * 2))

                        # Gesicht ausschneiden
                        face = imgwb[y:h, x:w]

                        # Hier die Logik für das Hinzufügen des Randes implementieren
                        # border_pixels = int(min(face.shape[0], face.shape[1]) * args.border)
                        #face_with_border = cv.copyMakeBorder(face, border_pixels, border_pixels, border_pixels, border_pixels, cv.BORDER_REFLECT)

                        # Entscheiden, ob das Bild dem Trainings- oder Validierungsordner zugewiesen wird
                        if random.uniform(0.0, 1.0) < args.split:
                            save_folder = person_val_folder
                        else:
                            save_folder = person_train_folder

                        # Check für Speicherordner
                        if not os.path.exists(save_folder):
                            os.makedirs(save_folder)

                        # Bild speichern
                        cv.imwrite(os.path.join(save_folder, image_file), face)

                        print(f"Face cropped from {image_file} and saved to {save_folder}")
                    
    if args.border is None:
        print("Cropping mode requires a border value to be set")
        exit()

    args.border = float(args.border)
    if args.border < 0 or args.border > 1:
        print("Border must be between 0 and 1")
        exit()