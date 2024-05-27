import cv2 as cv
import torch
import numpy as np
import argparse
import os
from PIL import Image
from network import Net
from transforms import ValidationTransform
import gdown
import uuid
import csv
from common import ROOT_FOLDER


def ensure_haar_cascade(xml_file_path):
    if not os.path.exists(xml_file_path):
        gdown.download(
            "https://drive.google.com/uc?id=1N5j5ke98qCt_0J70wg6F8diHrF5qqxeX&export=download",
            xml_file_path,
            quiet=False,
        )
        print("XML file downloaded successfully.")
    else:
        print("XML file already exists.")


def live(args):
    xml_file_path = "haarcascade_frontalface_default.xml"
    ensure_haar_cascade(xml_file_path)

    # Load the model checkpoint
    checkpoint = torch.load("model.pt", map_location=torch.device("cpu"))
    classes = checkpoint["classes"]
    model = Net(len(classes))
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Open webcam
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Load the face detection model
    face_cascade = cv.CascadeClassifier(xml_file_path)

    # Initialize the ValidationTransform
    validation_transform = ValidationTransform

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for x, y, w, h in faces:
            padding = args.border
            new_w = int(w * (1 + padding))
            new_h = int(h * (1 + padding))
            new_x = max(0, x - int((new_w - w) / 2))
            new_y = max(0, y - int((new_h - h) / 2))

            new_w = min(new_w, frame.shape[1] - new_x)
            new_h = min(new_h, frame.shape[0] - new_y)

            cv.rectangle(
                frame, (new_x, new_y), (new_x + new_w, new_y + new_h), (255, 0, 0), 2
            )

            face_with_border = frame[new_y : new_y + new_h, new_x : new_x + new_w]
            pil_image = Image.fromarray(cv.cvtColor(face_with_border, cv.COLOR_BGR2RGB))
            transformed_face = validation_transform(
                pil_image
            )  # Directly using the transform here

            with torch.no_grad():
                outputs = model(transformed_face.unsqueeze(0))
                _, predicted = torch.max(outputs, 1)
                label = classes[predicted.item()]

            cv.putText(
                frame,
                label,
                (new_x + 8, new_y - 6),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv.LINE_AA,
            )

        cv.imshow("Live Face Recognition", frame)

        if cv.waitKey(1) == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Live Face Recognition with Adjustable Border"
    )
    parser.add_argument(
        "--border",
        type=float,
        required=True,
        help="Percentage to increase the face bounding box",
    )
    args = parser.parse_args()
    live(args)
