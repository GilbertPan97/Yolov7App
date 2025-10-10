import os
import cv2
import argparse
import numpy as np

"""
YOLO Dataset Renderer
---------------------

This script visualizes YOLO-format datasets (bounding boxes or segmentation masks).

YOLO label formats:
    Detection (bbox): class_id x_center y_center width height
    Segmentation (mask): class_id x1 y1 x2 y2 ... xn yn

All coordinates are normalized (0~1).

Usage:
    python render_yolo.py --images <image_folder> --labels <label_folder> [--delay <milliseconds>] [--mode <type>]

Example:
    # Render detection dataset
    python render_yolo.py --images ./images --labels ./labels --mode bbox --delay 500

    # Render segmentation dataset
    python render_yolo.py --images ./images --labels ./labels --mode seg --delay 500

    python render_yolo.py --images ./workspace-box/yolo-cache/images/train --labels ./workspace-box/yolo-cache/labels/train --mode bbox --delay 500

Arguments:
    --images   Path to image folder (required)
    --labels   Path to YOLO label folder (required)
    --delay    Frame display delay in milliseconds (default: 500)
    --mode     Dataset type: 'bbox', 'seg', or 'auto' (default: auto)

Press ESC to exit early during rendering.
"""


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Render YOLO dataset with bounding boxes or segmentation masks")
    parser.add_argument("--images", type=str, required=True,
                        help="Path to image folder")
    parser.add_argument("--labels", type=str, required=True,
                        help="Path to YOLO labels folder")
    parser.add_argument("--delay", type=int, default=500,
                        help="Delay between frames in milliseconds (default: 500)")
    parser.add_argument("--mode", type=str, choices=["auto", "bbox", "seg"], default="auto",
                        help="Dataset type: 'bbox' (detection), 'seg' (segmentation), or 'auto' (default)")
    return parser.parse_args()


def load_yolo_labels(label_path, img_width, img_height, mode="auto"):
    """
    Load YOLO labels and convert them to pixel coordinates.
    Supports bbox and segmentation formats.
    """
    objects = []
    if not os.path.exists(label_path):
        return objects

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls = int(float(parts[0]))
            coords = list(map(float, parts[1:]))

            if mode == "bbox" or (mode == "auto" and len(coords) == 4):
                # Bounding box format
                x_center, y_center, w, h = coords
                x_center *= img_width
                y_center *= img_height
                w *= img_width
                h *= img_height
                x1 = int(x_center - w / 2)
                y1 = int(y_center - h / 2)
                x2 = int(x_center + w / 2)
                y2 = int(y_center + h / 2)
                objects.append(("bbox", cls, (x1, y1, x2, y2)))

            elif mode == "seg" or (mode == "auto" and len(coords) % 2 == 0):
                # Segmentation polygon format
                polygon = []
                for i in range(0, len(coords), 2):
                    x = int(coords[i] * img_width)
                    y = int(coords[i + 1] * img_height)
                    polygon.append((x, y))
                objects.append(("seg", cls, polygon))

    return objects


def resize_with_aspect_ratio(img, max_size=800):
    """Resize image keeping aspect ratio."""
    h, w = img.shape[:2]
    scale = min(max_size / w, max_size / h, 1.0)  # do not upscale
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def render_dataset(image_dir, label_dir, delay, mode):
    """Render YOLO dataset images with bounding boxes or segmentation masks"""
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue

        h, w = img.shape[:2]
        objects = load_yolo_labels(label_path, w, h, mode)

        # Draw objects
        for obj_type, cls, data in objects:
            if obj_type == "bbox":
                x1, y1, x2, y2 = data
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, str(cls), (x1, max(y1 - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif obj_type == "seg":
                pts = np.array(data, dtype=np.int32)
                cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

                # Semi-transparent fill
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], color=(0, 0, 255))
                cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

                cv2.putText(img, str(cls), (data[0][0], max(data[0][1] - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        img_resized = resize_with_aspect_ratio(img, max_size=800)
        cv2.imshow("YOLO Dataset Render", img_resized)
        key = cv2.waitKey(delay)
        if key == 27:  # ESC key
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    render_dataset(args.images, args.labels, args.delay, args.mode)
