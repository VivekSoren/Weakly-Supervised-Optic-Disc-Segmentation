import cv2 
import numpy as np
import os 
import pandas as pd
from tqdm import tqdm
from skimage.segmentation import slic
from skimage.util import img_as_float

from utils import calculate_metrics

def dice_score(pred, true):
    pred = pred.astype(bool)
    true = true.astype(bool)
    intersection = np.logical_and(pred, true).sum()
    return (2. * intersection) / (pred.sum() + true.sum() + 1e-8)

def process_image(img_path, mask_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = (gt_mask < 200).astype(np.uint8)
    # print(gt_mask.shape)

    # SLIC segmentation
    segments = slic(img_as_float(img_rgb), n_segments=600, compactness=10, start_label=1)

    # Score segments (brightness only)
    red = img_rgb[:, :, 0]
    scores = []
    for label in np.unique(segments):
        mask = (segments == label)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        x_mean, y_mean = np.mean(xs), np.mean(ys)
        brightness = np.mean(red[mask])
        scores.append((label, brightness, (x_mean, y_mean)))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_labels = [s[0] for s in scores[:8]]

    # Estimate OD center
    centroid_coords = np.array([s[2] for s in scores[:8]])
    cx, cy = int(np.mean(centroid_coords[:, 0])), int(np.mean(centroid_coords[:, 1]))

    # Raw mask
    raw_mask = np.isin(segments, top_labels).astype(np.uint8)

    # Fit ellipse around raw/weak mask
    contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ellipse_mask = np.zeros_like(raw_mask)
    ellipse = None

    if len(contours) > 0:
        largest = max(contours, key = cv2.contourArea)
        if len(largest) >= 5:
            ellipse = cv2.fitEllipse(largest)
            cv2.ellipse(ellipse_mask, ellipse, 1, -1)

    # Performance metrics
    # dice = dice_score(raw_mask, gt_mask)'
    metrics = calculate_metrics(raw_mask, gt_mask)
    return img_rgb, ellipse_mask, metrics, (cx, cy), ellipse, gt_mask

def crop_region(img, mask, crop_size=256, offset=100):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the binary image!")
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Fit a circle to the contour
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center_x, center_y = int(x), int(y)
    radius = int(radius)

    # Calculate crop coordinates
    half_size = crop_size // 2

    # Ensure crop stays withing image boundaries
    # print(img.shape)
    if len(img.shape) == 3:
        height, width, channel = img.shape
    else:
        height, width = img.shape
    start_x = max(0, center_x - half_size - offset)
    start_y = max(0, center_y - half_size - offset)
    end_x = min(width, center_x + half_size + offset)
    end_y = min(height, center_y + half_size + offset)
    # Crop the region
    cropped_region = img[start_y:end_y, start_x:end_x]

    # Ensure exact crop size
    if cropped_region.size == 0:
        print(start_x, start_y, end_x, end_y)
        raise ValueError(f'Cropped Region is empty')
    if cropped_region.shape != (crop_size, crop_size):
        cropped_region = cv2.resize(cropped_region, (crop_size, crop_size))
    circle_info = {
        'center': (center_x, center_y), 
        'radius': radius, 
        'crop_coords': (start_x, start_y, end_x, end_y), 
        'contour_area': cv2.contourArea(largest_contour)
    }
    return cropped_region, circle_info

if __name__ == '__main__':
    # SUBSETS = [f'train', f'val', f'test']
    SUBSETS = [f'val']
    for SUBSET in SUBSETS:
        IMAGES_DIR = os.path.join(f'REFUGE2/{SUBSET}', f'images')
        MASKS_DIR = os.path.join(f'REFUGE2/{SUBSET}', f'mask')
        SAVE_DIR = os.path.join(f'input/od', SUBSET)
        IMAGES_SAVE_DIR = os.path.join(SAVE_DIR, f'images')
        STRONG_SAVE_DIR = os.path.join(SAVE_DIR, f'strong_labels')
        WEAK_SAVE_DIR = os.path.join(SAVE_DIR, f'weak_labels')
        os.makedirs(IMAGES_SAVE_DIR, exist_ok=True)
        os.makedirs(STRONG_SAVE_DIR, exist_ok=True)
        os.makedirs(WEAK_SAVE_DIR, exist_ok=True)
        saved_flist = os.listdir(IMAGES_SAVE_DIR)

        # results = []
        dice_scores = []
        sensitivity_scores = []
        specificity_scores = []
        precision_scores = []
        for fname in tqdm(sorted(os.listdir(IMAGES_DIR)), total=len(os.listdir(IMAGES_DIR))):
            # if fname in saved_flist:
            #     continue
            img_path = os.path.join(IMAGES_DIR, fname)
            if SUBSET == f'val':
                mask_path = os.path.join(MASKS_DIR, fname.replace(".jpg", ".png"))
            else:
                mask_path = os.path.join(MASKS_DIR, fname.replace(".jpg", ".bmp"))

            img_rgb, ellipse_mask, metrics, center, ellipse, gt_mask = process_image(img_path, mask_path)
            
            # Crop mask
            cropped_mask, _ = crop_region(ellipse_mask, ellipse_mask, crop_size=256)
            cropped_img, _ = crop_region(img_rgb, ellipse_mask, crop_size=256)
            cropped_gt, _ = crop_region(gt_mask, ellipse_mask, crop_size=256)

            # Save the ellipse mask (weak label)
            # mask_path_save = os.path.join(SAVE_DIR, fname.replace(".jpg", "_ellipse_mask.png"))
            mask_path_save = os.path.join(WEAK_SAVE_DIR, fname)
            # cv2.imwrite(mask_path_save, cropped_mask * 255)
            # Save the image
            # image_path_save = os.path.join(SAVE_DIR, fname)
            image_path_save = os.path.join(IMAGES_SAVE_DIR, fname)
            # cv2.imwrite(image_path_save, cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
            # Save the gt mask
            # gt_path_save = os.path.join(SAVE_DIR, fname.replace(".jpg", "_strong_label.png"))
            gt_path_save = os.path.join(STRONG_SAVE_DIR, fname)
            # cv2.imwrite(gt_path_save, cropped_gt * 255)

            # Save overlay (ellipse + center)
            # overlay = img_rgb.copy()
            # if ellipse is not None:
            #     cv2.ellipse(overlay, ellipse, (255, 0, 0), 10)  # Red ellipse
            # cv2.circle(overlay, center, 5, (0, 255, 0), -1)     # Green center
            # overlay_save = os.path.join(SAVE_DIR, fname.replace(".jpg", f'_overlay_dice_{dice:.3f}.png'))
            # cv2.imwrite(overlay_save, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            # Save the performance metrics
            # results.append({"Image": fname, "Dice Score": dice})
            dice_scores.append(metrics['dice'])
            specificity_scores.append(metrics['specificity'])
            sensitivity_scores.append(metrics['sensitivity'])
            precision_scores.append(metrics['precision'])

            # print(img_path, mask_path)
            # break

        print(f'Samples in {IMAGES_SAVE_DIR}: {len(os.listdir(IMAGES_SAVE_DIR))}')
        print(f'Samples in {STRONG_SAVE_DIR}: {len(os.listdir(STRONG_SAVE_DIR))}')
        print(f'Samples in {WEAK_SAVE_DIR}: {len(os.listdir(WEAK_SAVE_DIR))}')
        print(f'Dice Score: {np.mean(dice_scores):.4f}, '
                f'Sensitivity: {np.mean(sensitivity_scores):.4f}, '
                f'Specificity: {np.mean(specificity_scores):.4f}, '
                f'Precision: {np.mean(precision_scores):.4f}')