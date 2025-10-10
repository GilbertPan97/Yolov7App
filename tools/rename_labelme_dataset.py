import os
import json
import shutil

def process_json_file(src_path, dst_path, suffix):
    """
    Process a labelme JSON file:
    - Update the imagePath field to include the suffix
    - Save the updated content to the new file path
    """
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "imagePath" in data:
        img_name, img_ext = os.path.splitext(data["imagePath"])
        data["imagePath"] = f"{img_name}{suffix}{img_ext}"

    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_regular_file(src_path, dst_path):
    """
    Copy a regular file (e.g., image or other resource) to the destination path
    """
    shutil.copy2(src_path, dst_path)


def add_suffix_and_save(src_folder, dst_folder, suffix="_v1"):
    """
    Iterate through all files in the source folder:
    - Add suffix to both image and JSON annotation files
    - Save modified files into the destination folder
    - Keep original files unchanged
    """
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    for filename in os.listdir(src_folder):
        src_path = os.path.join(src_folder, filename)

        if os.path.isdir(src_path):
            continue

        name, ext = os.path.splitext(filename)
        new_filename = f"{name}{suffix}{ext}"
        dst_path = os.path.join(dst_folder, new_filename)

        if ext.lower() == ".json":
            process_json_file(src_path, dst_path, suffix)
        else:
            process_regular_file(src_path, dst_path)

    print("All files saved with suffix into:", dst_folder)


if __name__ == "__main__":
    # Source folder containing original images and labelme JSONs
    src_folder = r"./dataset/Centus/RAW-backup/raw-14"
    # Destination folder to save renamed files
    dst_folder = r"./dataset/Centus/RAW/raw-14"
    add_suffix_and_save(src_folder, dst_folder, suffix="-14-centus")
