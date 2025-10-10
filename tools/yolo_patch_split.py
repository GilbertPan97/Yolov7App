import os
import cv2
import glob

def cut_image_with_labels(img_path, label_path, save_dir, dataset,
                          patch_size=1280, overlap=200, txt_file=None):
    """
    Cut a large image into patches and adjust YOLO labels accordingly.
    Also writes new patch paths to a txt file if provided.

    Args:
        img_path (str): Path to the large image
        label_path (str): Path to YOLO txt label
        save_dir (str): Output directory for patches
        dataset (str): Dataset type
        patch_size (int): Patch size (square)
        overlap (int): Overlap size between patches
        txt_file (file object): Open file to append new image paths
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image {img_path}")
        return
    h, w = img.shape[:2]

    # Load YOLO labels
    labels = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f.readlines():
                cls, x, y, bw, bh = line.strip().split()
                cls = int(cls)
                x, y, bw, bh = map(float, [x, y, bw, bh])

                # Convert normalized coordinates to absolute coordinates
                x1 = (x - bw / 2) * w
                y1 = (y - bh / 2) * h
                x2 = (x + bw / 2) * w
                y2 = (y + bh / 2) * h
                labels.append([cls, x1, y1, x2, y2])

    # Slide window over the image
    count = 0
    for y0 in range(0, h, patch_size - overlap):
        for x0 in range(0, w, patch_size - overlap):
            x1_patch = min(x0 + patch_size, w)
            y1_patch = min(y0 + patch_size, h)

            patch = img[y0:y1_patch, x0:x1_patch]
            ph, pw = patch.shape[:2]

            # Skip very small patches
            if ph < patch_size // 2 or pw < patch_size // 2:
                continue

            # Save patch image with original extension and max quality
            patch_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_{count}"
            ext = os.path.splitext(img_path)[1]  # keep original extension
            patch_img_path = os.path.join(save_dir, "images", dataset, patch_name + ext)
            patch_label_path = os.path.join(save_dir, "labels", dataset, patch_name + ".txt")

            # Ensure parent directory exists
            os.makedirs(os.path.dirname(patch_img_path), exist_ok=True)
            if ext.lower() in ['.jpg', '.jpeg']:
                cv2.imwrite(patch_img_path, patch, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            else:
                cv2.imwrite(patch_img_path, patch)

            # Write path to train.txt if provided
            if txt_file is not None:
                # Use relative path like ./images/train/xxx.png
                rel_path = os.path.relpath(patch_img_path, start=os.path.dirname(save_dir))
                rel_path = rel_path.replace("\\", "/")  # make it unix-style
                txt_file.write(f"./{rel_path}\n")
                txt_file.flush()

            # Adjust labels for this patch
            new_labels = []
            for cls, lx1, ly1, lx2, ly2 in labels:
                # Compute intersection with patch
                ix1 = max(lx1, x0)
                iy1 = max(ly1, y0)
                ix2 = min(lx2, x1_patch)
                iy2 = min(ly2, y1_patch)

                if ix1 < ix2 and iy1 < iy2:  # valid intersection
                    # Map to patch coordinates
                    px1 = ix1 - x0
                    py1 = iy1 - y0
                    px2 = ix2 - x0
                    py2 = iy2 - y0

                    # Convert back to YOLO normalized format
                    cx = (px1 + px2) / 2 / pw
                    cy = (py1 + py2) / 2 / ph
                    bw = (px2 - px1) / pw
                    bh = (py2 - py1) / ph

                    new_labels.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            # Save adjusted labels
            os.makedirs(os.path.dirname(patch_label_path), exist_ok=True)
            with open(patch_label_path, "w") as f:
                f.writelines(new_labels)

            count += 1

    print(f"Processed {img_path}, generated {count} patches.")


if __name__ == "__main__":
    # Base path
    base_dir = "./workspace-box/yolo"
    img_subdir = "images"     # Images folder
    label_subdir = "labels"   # Labels folder

    data_split = "test"        # Could be "train", "val", or "test"
    save_dir = "./workspace-box/yolo-cache/"

    # Construct full paths
    img_dir = os.path.join(base_dir, img_subdir, data_split)
    label_dir = os.path.join(base_dir, label_subdir, data_split)

    os.makedirs(save_dir, exist_ok=True)

    # Create or overwrite dataset txt
    train_txt_path = os.path.join(save_dir, f"{data_split}.txt")
    with open(train_txt_path, "w") as txt_file:

        # Recursively collect all images
        img_paths = glob.glob(os.path.join(img_dir, "**", "*.*"), recursive=True)

        for img_path in img_paths:
            # Relative path with respect to img_dir
            name_no_ext = os.path.splitext(os.path.basename(img_path))[0]

            # Corresponding label path
            label_path = os.path.join(label_dir, os.path.splitext(name_no_ext)[0] + ".txt")

            # Call your image cutting function
            cut_image_with_labels(
                img_path,
                label_path,
                save_dir,
                data_split,
                patch_size=1280,
                overlap=200,
                txt_file=txt_file
            )