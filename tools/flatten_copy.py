import os
import shutil


def copy_all_files(src_folder, dst_folder):
    """
    Recursively copy all files from src_folder (including subfolders)
    into dst_folder. The directory structure is not preserved.
    If a file with the same name already exists in dst_folder,
    print a red warning and skip copying.
    Show progress during copying.
    """
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # Collect all files first to count total
    all_files = []
    for root, _, files in os.walk(src_folder):
        for filename in files:
            all_files.append(os.path.join(root, filename))

    total = len(all_files)
    copied = 0

    for i, src_path in enumerate(all_files, start=1):
        filename = os.path.basename(src_path)
        dst_path = os.path.join(dst_folder, filename)

        if os.path.exists(dst_path):
            # Red warning for duplicates
            print(f"\033[91m[{i}/{total}] Warning: file '{filename}' already exists. Skipped.\033[0m")
        else:
            shutil.copy2(src_path, dst_path)
            copied += 1
            # Progress info
            percent = (i / total) * 100
            print(f"[{i}/{total}] Copied: {filename} ({percent:.2f}%)")

    print(f"\nCompleted: {copied}/{total} files copied successfully.")
    if copied < total:
        print(f"\033[91m{total - copied} files were skipped due to name conflicts.\033[0m")


if __name__ == "__main__":
    src_folder = r"./dataset/Centus/RAW"               # source folder
    dst_folder = r"./workspace-box/labelme"         # destination folder
    copy_all_files(src_folder, dst_folder)
