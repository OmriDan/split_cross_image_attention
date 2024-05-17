import os
from datetime import datetime
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def init(image_path='./images/two_cakes.jpeg'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_type = "vit_h"
    checkpoint_path = './sam_checkpoint/sam_vit_h_4b8939.pth'

    # Read the image from the path
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return device, model_type, checkpoint_path, image_rgb


def create_yolo_bbox(image_rgb):
    # Use YOLOv8 for bounding box creation
    model = YOLO('yolov8n.pt')
    yolo_results = model.predict(image_rgb)

    object_boxes = []
    boxes = yolo_results[0].boxes
    for idx, curr_box in enumerate(boxes):
        object_boxes.append(curr_box.xyxy)
    return object_boxes


def resize_masks(mask, sizes=[(32, 32), (64, 64)]):
    """
    Resizes a binary mask to the specified sizes.

    Args:
    mask (np.ndarray): A 2D numpy array representing the binary mask (boolean type).
    sizes (list of tuples): List of tuples indicating the sizes to resize to.

    Returns:
    dict: A dictionary with keys as size tuples and values as the resized masks.
    """
    # Convert boolean mask to uint8
    mask_uint8 = mask.astype(np.uint8) * 255

    resized_masks = {}
    for size in sizes:
        resized_mask = cv2.resize(mask_uint8, size, interpolation=cv2.INTER_NEAREST)
        # Convert back to boolean
        resized_masks[size] = torch.from_numpy(resized_mask == 255)
    return resized_masks


def create_sam_segmentation(image_path, n_objects=2, display=False):
    output_mask_dict_lst = []
    device, model_type, checkpoint_path, image_rgb = init(image_path)
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device=device)

    # Set up the SAM model with the encoded image
    mask_predictor = SamPredictor(sam)
    mask_predictor.set_image(image_rgb)

    # Use YOLOv8 for bounding box creation
    object_boxes = create_yolo_bbox(image_rgb)
    full_size_object_masks = []
    for object_box in object_boxes:
        # Predict mask with bounding box prompt
        masks, scores, logits = mask_predictor.predict(
            box=object_box.cpu().numpy(),
            multimask_output=True
        )
        chosen_mask = masks[np.argmax(scores)]
        full_size_object_masks.append(chosen_mask)
        if display:
            display_mask_and_bbox(image_rgb, object_box, chosen_mask)

        if len(full_size_object_masks) == n_objects:
            break
    for mask in full_size_object_masks:
        output_mask_dict_lst.append(resize_masks(mask))
    return output_mask_dict_lst


def sam_segmentation_flow(image_path, n_objects):
    print('Getting segmentations from yoloSAM flow...')
    return create_sam_segmentation(image_path, n_objects=n_objects)


def display_mask_and_bbox(image, bbox, mask):
    fig, ax = plt.subplots()
    ax.imshow(image)
    bbox = bbox.cpu().numpy()
    x_tl, y_tl, x_br, y_br = bbox[:,0], bbox[:,1], bbox[:,2], bbox[:,3]
    w = x_br - x_tl
    h = y_br - y_tl
    rect = patches.Rectangle((x_tl, y_tl), w, h, linewidth=2, edgecolor='r', facecolor='none')
    #ax.add_patch(rect)
    ax.imshow(np.asarray(mask), cmap='jet', alpha=0.5)  # Adjust alpha for transparency
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join("./masks_for_slides", f'overall_mask_{timestamp}')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    a=1
