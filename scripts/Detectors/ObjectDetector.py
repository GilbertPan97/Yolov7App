import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def model_loader(model_name, img_size=640):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize
    set_logging()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(model_name, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride

    trace = False       # Not trace model

    if trace:
        model = TracedModel(model, device, img_size)

    if half:
        model.half()  # to FP16

    return model


def inference(image, model, img_size=640, stride=32):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Padded resize
    img = letterbox(image, 640, stride=32, auto=True)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if device.type != "cpu" else img.float()

    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
        out = model(img)[0]

    out = non_max_suppression(out, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)

    im0 = image     # plot result image
    for i, det in enumerate(out):
        s = ''

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                # Add bbox to image
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

    return im0


if __name__ == '__main__':
    # Get the list of available cameras
    available_cameras = [f'Camera {i}' for i in range(3)]
    for camera_index, camera_name in enumerate(available_cameras):
        cap = cv2.VideoCapture(camera_index)

        # Check if the camera is available
        if cap.isOpened():
            print(f"{camera_name}: Available")
            cap.release()
        else:
            print(f"{camera_name}: Not Available")

    # Ask the user to select a camera to use
    camera_index = int(input("Enter the camera index you want to use: "))

    # Open the selected camera
    cap = cv2.VideoCapture(camera_index)

    # Check if the selected camera is successfully opened
    if not cap.isOpened():
        print("Could not open the selected camera")
        exit()

    # Load yolov7 model
    yolov7_model = model_loader('yolov7.pt')

    # Inference loop
    while True:
        # Read the camera image
        ret, frame = cap.read()

        img_result = inference(frame, yolov7_model)

        # Check if the image is successfully read
        if not ret:
            print("Could not read the image")
            break

        # Display the image in a window
        cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
        cv2.imshow('Camera', img_result)

        # Check if the 'q' key is pressed to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera resources and close the window
    cap.release()
    cv2.destroyAllWindows()