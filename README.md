# Yolov7App

## Overview

`Yolov7App` is a toolkit for object detection using the YOLOv7 model. It is derived from an open-source repository and is maintained to provide a complete workflow for data processing, model training, ONNX model conversion, and C++ deployment.

## Installation

1. **Clone the repository**

```
git clone https://github.com/GilbertPan97/Yolov7App
cd Yolov7App
```

2. **Install dependencies**

```
pip install -r requirements.txt
```

## Usage Manual

### 1. Data Conversion (Labelme to YOLO TXT Format)

Ensure your Labelme dataset is organized and use the provided script for conversion.

```
python tools/convert_labelme_to_txt.py --input_folder <input-folder> \
    --output_folder <output-folder> --label_map <categories-json> \
    --image_format <image-format> --train_ratio <train-ratio> --val_ratio <val-ratio> --test_ratio <test-ratio>
```

- `<input-folder>`: Directory containing images and LabelMe JSON annotation files.
- `<output-folder>`: Directory to save the converted YOLO TXT files.
- `<categories-json>`: Path to the category mapping JSON file.
- `<image-format>`: Image file format (e.g., png, jpg).
- `<train-ratio>`: Ratio of the dataset for training.
- `<val-ratio>`: Ratio of the dataset for validation.
- `<test-ratio>`: Ratio of the dataset for testing.

The script will generate `.txt` label files where each line corresponds to an object's class and bounding box in YOLO format.

### 2. Model Training

After preparing the data, start training with the following command:

```
python train.py --workers 8 --device 0 --batch-size 10 \
    --data <data-config.yaml> --img 640 640 \
    --cfg <model-config.yaml> --weights <initial-weights.pt> \
    --name <experiment-name> --hyp <hyperparameter-config.yaml>
```

- `<data-config.yaml>`: Path to the dataset configuration file.
- `<model-config.yaml>`: Path to the model configuration file.
- `<initial-weights.pt>`: Path to the initial weights (or use 'yolov7.pt' for pretrained weights).
- `<experiment-name>`: Name for the training experiment.
- `<hyperparameter-config.yaml>`: Path to the hyperparameter configuration file.

Example:

```
python train.py --workers 8 --device 0 --batch-size 10 \
    --data /media/gilbertpan/Elements/AlgaeDataset/WorkSpace/dataset_config.yaml --img 640 640 \
    --cfg cfg/training/yolov7.yaml --weights 'runs/train/box/weights/best.pt' \
    --name yolov7 --hyp data/hyp.scratch.p5.yaml
```

### 3. Model Conversion (to ONNX Format)

Once training is complete, convert the model to ONNX format for deployment.

```
python export.py --weights <trained-weights.pt>
```

Example:

```
python export.py --weights ./runs/train/yolov72/weights/best.pt
```

The exported ONNX model will be saved in the `runs/train/exp/weights` directory.

## Notes

- Ensure that the directory structure is correctly set up for training.
- Adjust hyperparameters in the configuration files as needed for your specific dataset.

## License

This project is licensed under the GNU General Public License.

## Acknowledgments

Special thanks to the original authors of the [YOLOv7 project](https://github.com/WongKinYiu/yolov7) for their outstanding work and contribution to the computer vision community.

## Contact

For any questions or issues, please contact gilbertpan97@gmail.com.