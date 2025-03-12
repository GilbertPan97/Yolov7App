import argparse
import cv2
import numpy as np
import os


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Render annotations onto an image.")
    parser.add_argument("--image", type=str, default="./coco/0a03abcc-3b12-46c5-98fc-4caec51623f8.png", help="Path to the input image.")
    parser.add_argument("--labels", type=str, default="./coco/0a03abcc-3b12-46c5-98fc-4caec51623f8.txt", help="Path to the label file.")
    parser.add_argument("--output", type=str, default="./out", help="Directory to save the output image.")
    parser.add_argument("--color", type=str, default="0,255,0", help="Color of the annotations (R,G,B).")
    parser.add_argument("--thickness", type=int, default=2, help="Line thickness for annotations.")
    return parser.parse_args()


def load_labels(label_path):
    """Load labels from the label file."""
    labels = []
    with open(label_path, "r") as f:
        for line in f:
            values = list(map(float, line.split()))
            category = int(values[0])
            points = [(values[i], values[i + 1]) for i in range(1, len(values), 2)]
            labels.append((category, points))
    return labels


def draw_annotations(image, labels, color, thickness):
    """Draw annotations on the image."""
    height, width = image.shape[:2]
    color = tuple(map(int, color.split(',')))
    for category, points in labels:
        scaled_points = [(int(x * width), int(y * height)) for x, y in points]
        cv2.polylines(image, [np.array(scaled_points, np.int32)], isClosed=True, color=color, thickness=thickness)
        centroid = np.mean(scaled_points, axis=0).astype(int)
        cv2.putText(image, str(category), tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
    return image


def main():
    args = parse_arguments()
    
    # Ensure the output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Read the input image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not open image {args.image}")
        return
    
    # Load labels and draw annotations
    labels = load_labels(args.labels)
    image = draw_annotations(image, labels, args.color, args.thickness)
    
    # Save the annotated image to the output directory
    output_path = os.path.join(args.output, os.path.basename(args.image))
    cv2.imwrite(output_path, image)
    print(f"Annotated image saved as {output_path}")


if __name__ == "__main__":
    main()