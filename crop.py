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
    """Process images by cropping faces and splitting into training and validation sets."""
    if args.border is None or not (0 <= float(args.border) <= 1):
        print("Cropping mode requires a border value between 0 and 1.")
        return
    
    # Convert border to float if passed as a string
    border_percentage = float(args.border)

    # Clean and prepare the output folders
    clean_folder(TRAIN_FOLDER)
    clean_folder(VAL_FOLDER)
    create_folders()

    # Process each person's folder in the root directory
    for person_folder in os.listdir(ROOT_FOLDER):
        person_folder_path = os.path.join(ROOT_FOLDER, person_folder)
        if os.path.isdir(person_folder_path):
            person_train_folder = os.path.join(TRAIN_FOLDER, person_folder)
            person_val_folder = os.path.join(VAL_FOLDER, person_folder)
            os.makedirs(person_train_folder, exist_ok=True)
            os.makedirs(person_val_folder, exist_ok=True)

            # Iterate over all image files in the person's folder
            for image_file in os.listdir(person_folder_path):
                if image_file.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(person_folder_path, image_file)
                    csv_file_path = os.path.join(person_folder_path, f"{os.path.splitext(image_file)[0]}.csv")

                    if not os.path.exists(csv_file_path):
                        continue

                    frame = cv.imread(image_path)
                    if frame is None:
                        continue

                    with open(csv_file_path, 'r') as csv_file:
                        csv_reader = csv.reader(csv_file)
                        for row in csv_reader:
                            x, y, w, h = map(int, row)
                            
                            # Calculate the number of pixels to add as border
                            border_pixels = int(border_percentage * min(w, h))

                            # Extend the frame with border to avoid out-of-bound errors when cropping
                            extended_frame = cv.copyMakeBorder(
                                frame, 
                                border_pixels, border_pixels, border_pixels, border_pixels, 
                                cv.BORDER_REFLECT
                            )

                            # Adjust coordinates due to the added border
                            x += border_pixels
                            y += border_pixels

                            # Crop the face with the border
                            face_with_border = extended_frame[
                                y - border_pixels : y + h + border_pixels,
                                x - border_pixels : x + w + border_pixels
                            ]

                            # Decide whether to save to training or validation folder
                            save_folder = person_val_folder if random.uniform(0.0, 1.0) < args.split else person_train_folder
                            cv.imwrite(os.path.join(save_folder, image_file), face_with_border)

                            print(f"Face cropped from {image_file} and saved to {save_folder}")
                    
    if args.border is None:
        print("Cropping mode requires a border value to be set")
        exit()

    args.border = float(args.border)
    if args.border < 0 or args.border > 1:
        print("Border must be between 0 and 1")
        exit()