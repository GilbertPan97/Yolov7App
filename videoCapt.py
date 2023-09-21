import cv2

import matplotlib.pyplot as plt
import torch
import yaml
from torchvision import transforms
import numpy as np

from utils.datasets import letterbox
from utils.general import non_max_suppression_mask_conf

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image


def inference():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open('data/hyp.scratch.mask.yaml') as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
    weigths = torch.load('yolov7-mask.pt')
    model = weigths['model']
    model = model.half().to(device) if device.type != "cpu" else model.float().to(device)
    _ = model.eval()

    image = cv2.imread('./inference/images/horses.jpg')  # 504x378 image
    image = letterbox(image, 640, stride=64, auto=True)[0]
    image_ = image.copy()
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    image = image.to(device)
    image = image.half() if device.type != "cpu" else image.float()
    output = model(image)

    inf_out, train_out, attn, mask_iou, bases, sem_output = output['test'], output['bbox_and_cls'], output['attn'], \
    output['mask_iou'], output['bases'], output['sem']

    bases = torch.cat([bases, sem_output], dim=1)
    nb, _, height, width = image.shape
    names = model.names
    pooler_scale = model.pooler_scale
    pooler = ROIPooler(output_size=hyp['mask_resolution'], scales=(pooler_scale,), sampling_ratio=1,
                       pooler_type='ROIAlignV2', canonical_level=2)

    output, output_mask, output_mask_score, output_ac, output_ab = \
        non_max_suppression_mask_conf(inf_out, attn, bases, pooler, hyp, conf_thres=0.25, iou_thres=0.65, merge=False, mask_iou=None)

    pred, pred_masks = output[0], output_mask[0]
    base = bases[0]
    bboxes = Boxes(pred[:, :4])
    original_pred_masks = pred_masks.view(-1, hyp['mask_resolution'], hyp['mask_resolution'])
    pred_masks = retry_if_cuda_oom(paste_masks_in_image)(original_pred_masks, bboxes, (height, width), threshold=0.5)
    pred_masks_np = pred_masks.detach().cpu().numpy()
    pred_cls = pred[:, 5].detach().cpu().numpy()
    pred_conf = pred[:, 4].detach().cpu().numpy()
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    nbboxes = bboxes.tensor.detach().cpu().numpy().astype(np.int)
    pnimg = nimg.copy()

    for one_mask, bbox, cls, conf in zip(pred_masks_np, nbboxes, pred_cls, pred_conf):
        if conf < 0.25:
            continue
        color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]

        pnimg[one_mask] = pnimg[one_mask] * 0.5 + np.array(color, dtype=np.uint8) * 0.5
        pnimg = cv2.rectangle(pnimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        # label = '%s %.3f' % (names[int(cls)], conf)
        # t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
        # c2 = bbox[0] + t_size[0], bbox[1] - t_size[1] - 3
        # pnimg = cv2.rectangle(pnimg, (bbox[0], bbox[1]), c2, color, -1, cv2.LINE_AA)  # filled
        # pnimg = cv2.putText(pnimg, label, (bbox[0], bbox[1] - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    # coco example
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(pnimg)
    plt.show()


if __name__ == '__main__':
    # Get the list of available cameras
    inference()

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

    while True:
        # Read the camera image
        ret, frame = cap.read()

        # Check if the image is successfully read
        if not ret:
            print("Could not read the image")
            break

        # Display the image in a window
        cv2.imshow('Camera', frame)

        # Check if the 'q' key is pressed to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera resources and close the window
    cap.release()
    cv2.destroyAllWindows()

