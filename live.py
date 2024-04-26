import cv2 as cv
import torch
import os
from network import Net
#from cascade import create_cascade
from transforms import ValidationTransform
from PIL import Image

# NOTE: This will be the live execution of your pipeline

def live(args):
    # Load the model checkpoint from a previous training session
    checkpoint = torch.load("model.pt")
    classes = checkpoint["classes"]
    model = Net(len(classes))
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Create a video capture device to retrieve live footage from the webcam
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Couldn't open camera")
        exit()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Couldn't retrieve frame from camera")
            break

        # Attach border to the whole video frame for later cropping
        border_pixels = int(min(frame.shape[0], frame.shape[1]) * args.border)
        frame_with_border = cv.copyMakeBorder(frame, border_pixels, border_pixels, border_pixels, border_pixels, cv.BORDER_REFLECT)

        # Show the frame
        cv.imshow('Live Face Recognition', frame)

        if cv.waitKey(1) & 0xFF == ord('q'): # q fpr exit
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    args = None  # You need to define the args or parse command line arguments here
    live(args)
