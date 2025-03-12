import os
import argparse
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import time
import yaml
from typing import List
from glob import glob

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933], [0.941, 0.972, 1.000]]

def normalize_color(color):
    """
    Convert normalized [0, 1] color to PIL-compatible RGB integer tuple.
    """
    return tuple(int(ch * 255) for ch in color)

def read_yaml_names(yaml_path: str) -> List[str]:
    """Reads the 'names' field from a YAML file and returns it as a list of strings"""
    with open(yaml_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)  # Parse the YAML file
    return data.get("names", [])  # Return the 'names' field, or an empty list if it doesn't exist

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape using numpy
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2,
               (img1_shape[0] - img0_shape[0] * gain) / 2)  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def scale_coords1(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape using numpy
    if ratio_pad is None:  # Calculate scaling ratio based on img0_shape
        # Separate scaling factors for width (X) and height (Y)
        gain_x = img0_shape[1] / img1_shape[1]  # X-direction scaling factor (width)
        gain_y = img0_shape[0] / img1_shape[0]  # Y-direction scaling factor (height)
    else:
        # Use provided scaling ratio if available
        gain_x = ratio_pad[0][0]
        gain_y = ratio_pad[0][1]

    # Scale coordinates independently for X and Y directions
    coords[:, [0, 2]] *= gain_x  # Scale X coordinates (left, right)
    coords[:, [1, 3]] *= gain_y  # Scale Y coordinates (top, bottom)

    # Clip coordinates to ensure they are within the bounds of the original image
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width) using numpy
    boxes[:, 0] = np.clip(boxes[:, 0], 0, img_shape[1])  # x1
    boxes[:, 1] = np.clip(boxes[:, 1], 0, img_shape[0])  # y1
    boxes[:, 2] = np.clip(boxes[:, 2], 0, img_shape[1])  # x2
    boxes[:, 3] = np.clip(boxes[:, 3], 0, img_shape[0])  # y2

def xywh2xyxy(x):
    """Convert [x, y, w, h] to [x1, y1, x2, y2]"""
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y


def iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes"""
    x1 = np.maximum(boxes1[:, None, 0], boxes2[:, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[:, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[:, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[:, 3])

    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    box2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = box1_area[:, None] + box2_area - inter_area

    return inter_area / (union_area + 1e-6)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False,
                              multi_label=False, labels=()):
    """Runs Non-Maximum Suppression (NMS) using numpy"""
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # confidence threshold mask

    min_wh, max_wh = 2, 4096  # minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into NMS
    time_limit = 10.0  # seconds to quit after
    t = time.time()
    output = [np.zeros((0, 6))] * prediction.shape[0]

    for xi, x in enumerate(prediction):
        x = x[xc[xi]]  # filter by confidence
        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        box = xywh2xyxy(x[:, :4])

        conf = x[:, 5:].max(1, keepdims=True)
        j = x[:, 5:].argmax(1, keepdims=True)
        x = np.concatenate((box, conf, j.astype(np.float32)), axis=1)

        if classes is not None:
            x = x[np.isin(x[:, 5], classes)]

        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort()[::-1][:max_nms]]

        boxes, scores = x[:, :4], x[:, 4]

        keep = []
        while len(x) > 0:
            keep.append(x[0])
            if len(x) == 1:
                break
            ious = iou(x[0:1, :4], x[1:, :4]).flatten()
            x = x[1:][ious < iou_thres]

            if len(keep) >= max_det:
                break

        output[xi] = np.array(keep)

        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break

    return output


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def preprocess_image(image_path, img_size=(640, 640)):
    """
    Load and preprocess an image for ONNX inference.
    Args:
        image_path (str): Path to the input image.
        img_size (tuple): Target image size (width, height).
    Returns:
        np.ndarray: Preprocessed image tensor.
        tuple: Original image shape (height, width).
    """
    img = cv2.imread(image_path)
    orig_h, orig_w = img.shape[:2]  # Get original image dimensions

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, img_size)  # Resize to the required model input size
    # img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    # img = np.transpose(img, (2, 0, 1))  # Convert to (C, H, W)
    #
    # img_tensor = np.expand_dims(img, axis=0)  # Add batch dimension

    # Padded resize
    img = letterbox(img, auto=False)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
    img = np.ascontiguousarray(img)
    # Ensure the image is in float type before performing arithmetic
    img = img.astype(np.float32)  # Convert to float32
    img /= 255.0        # Normalize to [0, 1]

    img_tensor = np.expand_dims(img, axis=0)

    return img_tensor, (orig_h, orig_w)

def postprocess(pred, img_shape, conf_thres=0.3, iou_thres=0.45):
    """
    Post-process YOLOv7 output, including applying NMS.
    Args:
        pred (np.ndarray): Raw model output tensor (1, 25200, 85).
        img_shape (tuple): Original image shape (height, width).
        conf_thres (float): Confidence threshold.
        iou_thres (float): IoU threshold for NMS.
    Returns:
        list: Final bounding boxes after NMS.
        list: Confidence scores after NMS.
        list: Class IDs after NMS.
    """
    orig_h, orig_w = img_shape

    # Apply Non-Maximum Suppression (NMS)
    detections = non_max_suppression(pred[0], conf_thres, iou_thres)

    det_boxes, det_scores, det_classes = [], [], []
    for det in detections:
        if det is not None and len(det):
            # Rescale boxes from [0, 1] to original image shape
            det[:, :4] = scale_coords((640, 640), det[:, :4], (orig_h, orig_w)).round()

            for *xyxy, conf, cls in det:
                det_boxes.append(xyxy)
                det_scores.append(conf.item())
                det_classes.append(int(cls.item()))

    return det_boxes, det_scores, det_classes


def bbox_renderer(pil_img, prob, classes, boxes, categories, display=False, save_path=None, standard_width=1280):
    """
    Render bounding boxes on an image with labels and confidence scores.

    Args:
        pil_img (PIL.Image.Image or np.ndarray): The input image to render bounding boxes on.
        prob (list): Probability list with shape (N), where N is the number of bounding boxes.
        classes (list of int): List of class index.
        boxes (list): Bounding boxes list with shape (N), where each box is (xmin, ymin, xmax, ymax).
        display (bool): Whether to display the resulting image using PIL's show method.
        save_path (str or None): Path to save the resulting image. If None, the image is not saved.
        standard_width (int): Standard width to calculate the scale factor based on image width.
    """

    # Convert image to RGB if not already
    if isinstance(pil_img, np.ndarray):
        pil_img = Image.fromarray(pil_img)
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')

    # Get image size and calculate scale factor
    image_width, image_height = pil_img.size
    scaleFactor = image_width / standard_width

    # Create a drawing context
    draw = ImageDraw.Draw(pil_img)

    # Define a font for text
    try:
        font_size = int(20 * scaleFactor)  # Adjust font size according to scale factor
        font = ImageFont.truetype("DejaVuSans.ttf", size=font_size)
    except IOError:
        font = ImageFont.load_default()

    # Draw bounding boxes and labels
    num_boxes = len(boxes)

    if num_boxes > len(COLORS):
        raise ValueError("Not enough colors provided for the number of bounding boxes.")

    for i, (p, cl, (xmin, ymin, xmax, ymax), color) in enumerate(zip(prob, classes, boxes, COLORS)):
        try:
            # Convert color from normalized [0, 1] to RGB integer tuple
            color = normalize_color(color)

            # Draw rectangle with scaled line width
            line_width = int(3 * scaleFactor)  # Adjust line width according to scale factor
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=line_width)

            # Prepare label text
            text = f'{categories[cl]}: {p:0.2f}'

            # Draw text background and text
            text_bbox = draw.textbbox((xmin, ymin), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            padding = int(10 * scaleFactor)  # Adjust padding according to scale factor

            # Adjust text background position and size
            y_offset = padding
            text_bg_xmin = xmin
            text_bg_ymin = ymin - text_height - 2 * padding - y_offset  # Ensure there's space above the bounding box
            text_bg_xmax = xmin + text_width + 2 * padding
            text_bg_ymax = ymin - y_offset

            # Draw background rectangle with border
            draw.rectangle([text_bg_xmin, text_bg_ymin, text_bg_xmax, text_bg_ymax], fill='yellow', outline='black',
                           width=int(2 * scaleFactor))

            # Calculate text position to be vertically centered
            text_x = xmin + padding
            text_y = ymin - text_height - padding - (text_height // 2)
            # Center text vertically within the background rectangle
            text_y = (text_bg_ymin + text_bg_ymax - text_height) // 2

            draw.text((text_x, text_y), text, fill='black', font=font)
        except Exception as e:
            print(f"Error drawing box at ({xmin}, {ymin}, {xmax}, {ymax}): {e}")

    # Save or display image
    if save_path:
        pil_img.save(save_path)

    if display:
        pil_img.show()

    return pil_img

def print_ndarray_data(ndarray):
    # Check if the ndarray is 3D (channels x height x width)
    if len(ndarray.shape) != 3 or ndarray.shape[0] != 3:
        print("Error: The input ndarray must be a 3-channel image (channels x height x width)")
        return

    # Print the matrix data (values of each pixel)
    # for i in range(ndarray.shape[1]):  # Loop over rows (height)
    #     for j in range(ndarray.shape[2]):  # Loop over columns (width)
    #         # Print pixel data for all three channels (R, G, B)
    #         r, g, b = ndarray[0, i, j], ndarray[1, i, j], ndarray[2, i, j]
    #         print(f"Pixel ({i}, {j}): R={r}, G={g}, B={b}")  # Print the R, G, B values


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="YOLOv7 ONNX model inference with NMS.")
    parser.add_argument("--model", type=str, default="./runs/train/box/weights/best.onnx", help="Path to the ONNX model.")
    parser.add_argument("--data", type=str, default="data/customer.yaml", help="Path to the data.yaml file.")
    parser.add_argument("--images", type=str, default="/media/gilbertpan/Elements/AlgaeDataset/validate/test/",
                        help="Path to the folder containing input images.")

    # parser.add_argument("--model", type=str, default="weights/yolov7.onnx", help="Path to the ONNX model.")
    # parser.add_argument("--data", type=str, default="data/coco.yaml", help="Path to the data.yaml file.")
    # parser.add_argument("--images", type=str, default="inference/images/", help="Path to the folder containing input images.")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold for detections.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS.")
    args = parser.parse_args()

    # Load ONNX model
    # Check available execution providers
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        selected_provider = ["CUDAExecutionProvider"]
        print("Using GPU (CUDA) for inference.")
    else:
        selected_provider = ["CPUExecutionProvider"]
        print("CUDA not available. Using CPU for inference.")
    ort_session = ort.InferenceSession(args.model, providers=selected_provider)

    # Get model input and output names
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name  # The only output, shape (1, 25200, 85)

    # Get all image files from the folder
    image_paths = sorted(glob(os.path.join(args.images, "*.[jpJP][pnPN]*[gG]")))  # Supports .jpg, .jpeg, .png

    if not image_paths:
        print(f"No images found in {args.images}")
        return

    categories = read_yaml_names(args.data)

    for image_path in image_paths:
        print(f"Processing: {image_path}")

        # Preprocess the input image
        input_tensor, img_shape = preprocess_image(image_path)

        print_ndarray_data(input_tensor[0])

        # Perform inference
        outputs = ort_session.run([output_name], {input_name: input_tensor})

        # Postprocess (apply NMS)
        det_boxes, det_scores, det_classes_id = postprocess(outputs, img_shape, args.conf, args.iou)

        # Print results
        print(f"Detected {len(det_boxes)} objects in {image_path}")

        # Ensure output directory exists
        save_dir = os.path.join(args.images, "runs")
        os.makedirs(save_dir, exist_ok=True)  # Create 'runs' folder if it doesn't exist

        # Extract the filename and create the save path
        filename = os.path.basename(image_path)  # Extracts 'image1.jpg' from 'inference/images/image1.jpg'
        save_file = os.path.join(save_dir, filename)  # 'runs/image1.jpg'

        # Save the processed image
        bbox_renderer(Image.open(image_path), det_scores, det_classes_id, det_boxes, categories, display=False, save_path=save_file)
        #
        # img = input_tensor[0].astype(np.float32) * 255.0  # Normalize to [0, 1]
        # img = np.transpose(img, (2, 0, 1))  # Convert to (W, H, C)
        # bbox_renderer(img, det_scores, det_classes_id, det_boxes, categories, display=False,
        #               save_path=save_file)

if __name__ == "__main__":
    main()
