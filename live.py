import cv2 as cv
import torch
import os
from network import Net
#from cascade import create_cascade
from transforms import ValidationTransform
from PIL import Image

# NOTE: This will be the live execution of your pipeline

def live(args):
    # TODO: 
    #   Load the model checkpoint from a previous training session (check code in train.py)
    #   Initialize the face recognition cascade again (reuse code if possible)
    #   Also, create a video capture device to retrieve live footage from the webcam.
    #   Attach border to the whole video frame for later cropping.
    #   Run the cascade on each image, crop all faces with border.
    #   Run each cropped face through the network to get a class prediction.
    #   Retrieve the predicted persons name from the checkpoint and display it in the image
    if args.border is None:
        print("Live mode requires a border value to be set")
        exit()

    args.border = float(args.border)
    if args.border < 0 or args.border > 1:
        print("Border must be between 0 and 1")
        exit()