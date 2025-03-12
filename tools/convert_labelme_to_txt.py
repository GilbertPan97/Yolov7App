import json
import os
import argparse
import random
import shutil
import yaml

"""
Script to convert LabelMe JSON annotations to TXT format and split the dataset into train/val/test sets.

Usage:
    python convert_labelme_to_txt.py --input_folder <path_to_labelme_json_folder> \
    --output_folder <path_to_output_dir> --label_map <path_to_label_map_json> \
    --image_extension <image_format> --train_ratio 0.7 --val_ratio 0.1 --test_ratio 0.2

Arguments:
    --input_folder    : Path to the folder containing LabelMe JSON files.
    --output_folder   : Directory to save the output datasets (TXT files and images).
    --label_map       : Path to the label mapping JSON file.
    --image_extension : Format of the image files (default: png).
    --train_ratio     : Ratio of the dataset to use for training (default: 0.7).
    --val_ratio       : Ratio of the dataset to use for validation (default: 0.1).
    --test_ratio      : Ratio of the dataset to use for testing (default: 0.2).

Example:
    python tools/convert_labelme_to_txt.py --input_folder /path/to/labelme_images_jsons \
    --output_folder /path/to/output --label_map /path/to/categories.json \
    --image_extension png --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
    
    python tools/convert_labelme_to_txt.py --input_folder /media/gilbertpan/Elements/AlgaeDataset/labelme \
    --output_folder /media/gilbertpan/Elements/AlgaeDataset/WorkSpace \
    --label_map /media/gilbertpan/Elements/AlgaeDataset/WorkSpace/categories.json \
    --image_extension png --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1

Notes:
    - The sum of train_ratio, val_ratio, and test_ratio must equal 1.0.
    - You can adjust the ratios as needed for your specific dataset split.
"""

def load_label_map(json_path):
    """Load a category-to-index mapping from a given JSON file."""
    with open(json_path, 'r') as f:
        categories = json.load(f)
    label_map = {v: int(k) for k, v in categories.items()}
    return label_map


def convert_labelme_to_txt(input_json, output_txt, label_map, image_width, image_height):
    """Convert LabelMe JSON annotation to TXT format with coordinates as percentages."""
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(output_txt, 'w', encoding='utf-8') as f:
        for shape in data.get('shapes', []):
            # If the label is invalid, skip this annotation instance
            label = label_map.get(shape.get('label', '0'), -1)
            if label == -1:
                continue  # Skip current annotation instance

            points = shape.get('points', [])
            points_str = ' '.join(f"{(x / image_width):.6f} {(y / image_height):.6f}" for x, y in points)
            f.write(f"{label} {points_str}\n")


def process_folder(input_folder, output_folder, label_map, ratios, image_extension):
    """Process all LabelMe JSON files in a folder, convert them to TXT format, and split into train/test/val."""
    os.makedirs(output_folder, exist_ok=True)
    labels_folder = os.path.join(output_folder, 'labels')
    images_folder = os.path.join(output_folder, 'images')

    # Create subdirectories for train, test, and val
    for subfolder in ['train', 'test', 'val']:
        os.makedirs(os.path.join(labels_folder, subfolder), exist_ok=True)
        os.makedirs(os.path.join(images_folder, subfolder), exist_ok=True)

    json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    random.shuffle(json_files)

    # Split into train, test, and val
    total_files = len(json_files)
    train_end = int(total_files * ratios['train'])
    test_end = train_end + int(total_files * ratios['test'])

    train_files = json_files[:train_end]
    test_files = json_files[train_end:test_end]
    val_files = json_files[test_end:]

    # Prepare to record image paths
    train_images = []
    test_images = []
    val_images = []

    # Process each file and write converted data
    for file_list, subfolder, image_paths in zip([train_files, test_files, val_files],
                                                 ['train', 'test', 'val'],
                                                 [train_images, test_images, val_images]):
        for file_name in file_list:
            input_json = os.path.join(input_folder, file_name)
            output_txt = os.path.join(labels_folder, subfolder, file_name.replace('.json', '.txt'))

            # Read image width and height
            with open(input_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            image_width = data['imageWidth']
            image_height = data['imageHeight']

            # Convert labelme to txt
            convert_labelme_to_txt(input_json, output_txt, label_map, image_width, image_height)

            # Record image path relative to the root
            image_paths.append(f"./images/{subfolder}/{file_name.replace('.json', '.' + image_extension)}")
            print(f"Converted {input_json} to {output_txt}")

            # Copy the image to the appropriate folder
            image_file = file_name.replace('.json', '.' + image_extension)
            image_src = os.path.join(input_folder, image_file)
            image_dst = os.path.join(images_folder, subfolder, image_file)
            shutil.copy(image_src, image_dst)

    # Write paths to respective txt files
    for image_paths, filename in zip([train_images, test_images, val_images],
                                     ['train.txt', 'test.txt', 'val.txt']):
        with open(os.path.join(output_folder, filename), 'w', encoding='utf-8') as f:
            for path in image_paths:
                f.write(path + '\n')


def generate_yaml(output_folder, label_map):
    """Generate the YAML file for dataset configuration manually, dynamically setting the number of classes."""

    # Define the content for the YAML file
    yaml_content = """# train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
train: ./train.txt  # number of training images
val: ./val.txt      # number of validation images
test: ./test.txt    # number of test images

# number of classes
nc: {0}

# class names
names: [ """

    # Add class names to the content
    class_names = [f"'{name}'" for name in label_map]
    yaml_content += ", ".join(class_names)

    # Closing the names list and the YAML content
    yaml_content += " ]"

    # Replace {0} with the actual number of classes in the label_map
    yaml_content = yaml_content.format(len(label_map))

    # Write the content to the file
    yaml_file = os.path.join(output_folder, 'dataset_config.yaml')
    with open(yaml_file, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"Generated YAML configuration file: {yaml_file}")


def main():
    """Main function to parse arguments and run the conversion."""
    parser = argparse.ArgumentParser(description="Convert LabelMe JSON to TXT format and split into datasets.")
    parser.add_argument('--input_folder', type=str, help="Path to the folder containing images and LabelMe JSON files")
    parser.add_argument('--output_folder', type=str, help="Path to the folder to save output TXT files")
    parser.add_argument('--label_map', type=str, help="Path to the label mapping JSON file")
    parser.add_argument('--train_ratio', type=float, default=0.7, help="Ratio of training set")
    parser.add_argument('--test_ratio', type=float, default=0.2, help="Ratio of test set")
    parser.add_argument('--val_ratio', type=float, default=0.1, help="Ratio of validation set")
    parser.add_argument('--image_extension', type=str, default='png', help="Image file extension")

    args = parser.parse_args()

    # Ensure ratios sum to 1
    total_ratio = args.train_ratio + args.test_ratio + args.val_ratio
    if not (abs(total_ratio - 1.0) < 1e-5):
        raise ValueError(f"The sum of train_ratio, test_ratio, and val_ratio must be 1 (current sum: {total_ratio}).")

    label_map = load_label_map(args.label_map)
    ratios = {'train': args.train_ratio, 'test': args.test_ratio, 'val': args.val_ratio}
    process_folder(args.input_folder, args.output_folder, label_map, ratios, args.image_extension)

    # Generate the YAML file for dataset configuration
    generate_yaml(args.output_folder, label_map)


if __name__ == '__main__':
    main()