import cv2

import matplotlib.pyplot as plt
import torch
import yaml
from torchvision import transforms
import numpy as np

from utils.datasets import letterbox
from utils.general import non_max_suppression_mask_conf
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image


def model_loader(model_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weigths = torch.load(model_name)
    model = weigths['model']
    _ = model.float().eval()
    model = model.half().to(device) if device.type != "cpu" else model.float().to(device)

    return model


def inference(image, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image = letterbox(image, 640, stride=64, auto=True)[0]
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))

    image = image.half().to(device) if device.type != "cpu" else image.float().to(device)
    output, res = model(image)

    output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)

    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

    return nimg


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
    yolov7_model = model_loader('yolov7-w6-pose.pt')

    # Inference loop
    while True:
        # Read the camera image
        ret, frame = cap.read()

        with torch.no_grad():
            img_result = inference(frame, yolov7_model)

        # Check if the image is successfully read
        if not ret:
            print("Could not read the image")
            break

        # Display the image in a window
        cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
        cv2.imshow('Camera', img_result)

        # Check if the 'q' key is pressed to exit the loop
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release the camera resources and close the window
    cap.release()
    cv2.destroyAllWindows()
