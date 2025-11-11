import os, time, random
import numpy as np 
import cv2 
import pandas as pd 
from tqdm import tqdm

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader 
import torchvision.transforms as T 

from dataset import *
from model import *
from utils import * 

if __name__ == '__main__':
    OUT_DIR = f'results'
    INPUT_DIR = f'input/od'
    IMG_SIZE = (256, 256)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 8
    EPOCHS = 100 
    LR = 1e-4
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    CHECKPOINT_DIR = os.path.join(OUT_DIR, f'checkpoints')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    MODEL_PATH = os.path.join(CHECKPOINT_DIR, "unet.pth")

    # img_path = os.path.join(INPUT_DIR, f'images')
    # strong_path = os.path.join(INPUT_DIR, f'strong_labels')
    # weak_path = os.path.join(INPUT_DIR, f'weak_labels')
    # img_list = os.listdir(img_path)

    # train_ds = DiscDataset(img_path, weak_path, img_size=IMG_SIZE, augment=True)
    # test_ds = DiscDataset(img_path, strong_path, img_size=IMG_SIZE, augment=False)
    
    # Create dataset
    train_path = os.path.join(INPUT_DIR, f'train')
    img_path = os.path.join(train_path, f'images')
    weak_path = os.path.join(train_path, f'strong_labels')
    train_list = os.listdir(img_path)
    train_ds = DiscDataset(img_path, weak_path, img_size=IMG_SIZE, augment=True)

    valid_path = os.path.join(INPUT_DIR, f'val')
    img_path = os.path.join(valid_path, f'images')
    weak_path = os.path.join(valid_path, f'strong_labels')
    valid_list = os.listdir(img_path)
    valid_ds = DiscDataset(img_path, weak_path, img_size=IMG_SIZE, augment=False)

    test_path = os.path.join(INPUT_DIR, f'val')
    img_path = os.path.join(test_path, f'images')
    strong_path = os.path.join(test_path, f'strong_labels')
    weak_path = os.path.join(test_path, f'weak_labels')
    test_list = os.listdir(img_path)
    test_ds = CombinedDiscDataset(img_path, strong_path, weak_path, img_size=IMG_SIZE, augment=False)
    # test_ds = DiscDataset(img_path, strong_path, img_size=IMG_SIZE, augment=False)

    # Compute the Dice Score between the strong labels and weak labels
    num_samples = len(train_ds)
    dice_scores = []
    sensitivity_scores = []
    specificity_scores = []
    precision_scores = []
    for idx in range(num_samples):
        try:
            # print(f'Processing sample {idx}: \n')
            _, s_mask, w_mask = test_ds[idx]
            # print(s_mask.shape, w_mask.shape)
            # _, s_mask = test_ds[idx]
            # Calculate dice score
            # dice = dice_score(w_mask, s_mask)
            metrics = calculate_metrics(w_mask, s_mask)
            dice_scores.append(metrics['dice'])
            sensitivity_scores.append(metrics['sensitivity'])
            specificity_scores.append(metrics['specificity'])
            precision_scores.append(metrics['precision'])
        except Exception as e: 
            print(f'Error processing sample {idx}: {e}')
            continue
        # break
    # dice_scores = np.array(dice_scores)
    print(f'Dice score between strong labels and weak labels: {np.mean(dice_scores):.4f}, '
          f'Sensitivity: {np.mean(sensitivity_scores):.4f}, '
          f'Specificity: {np.mean(specificity_scores):.4f}, '
          f'Precision: {np.mean(precision_scores):.4f}')
                    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    for idx, item in enumerate(train_loader):
        print(f'Image Batch Size: {item[0].size()}')
        print(f'Mask Batch Size: {item[1].size()}')
        break 

    model = UNetDisc(in_channels=3, out_channels=1, base=32).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3, min_lr=1e-7)

    best_val = -1.0
    if os.path.exists(MODEL_PATH):
        print(f'Loading checkpoint from {MODEL_PATH}...')
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val = checkpoint('best_val')
        start_epoch = checkpoint['epoch'] + 1
        print(f"Model loaded from {MODEL_PATH}, Best Dice: {best_val:.4f}")
    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss = 0.0
        n = 0
        t0 = time.time()
        for img, mask in tqdm(train_loader, total=len(train_loader)):
            opt.zero_grad()
            # print(img.size(), mask.size())
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            logits = model(img)
            loss_bce = F.binary_cross_entropy_with_logits(logits.squeeze(1), mask)
            loss_dice = dice_loss_logits(logits, mask)
            loss = loss_bce + loss_dice
            loss.backward()
            opt.step()
            running_loss += float(loss.item())
            n += 1
        t1 = time.time()
        avg_loss = running_loss / max(1, n)

        # Testing against the strong labels
        model.eval()
        # dices = []
        val_metrics = {
            'dice': [], 'sensitivity': [], 'specificity': [], 'precision': [], 
            'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0
        }
        with torch.no_grad():
            for img, mask in valid_loader:
                img = img.to(DEVICE)
                logits = model(img)
                probs = torch.sigmoid(logits).cpu().numpy()[:, 0]
                preds = (probs > 0.5).astype(np.uint8)
                mask_np = mask.numpy().astype(np.uint8)
                # Calculate metrics for this batch
                batch_metrics = calculate_metrics_batch(preds, mask_np)
                # Accumulate metrics
                for key in ['dice', 'sensitivity', 'specificity', 'precision']:
                    val_metrics[key].append(batch_metrics[key])
                # Accumulate confusion matrix components
                for key in ['tp', 'fp', 'fn', 'tn']:
                    val_metrics[key] += batch_metrics[key]
        val_dice = float(np.mean(val_metrics['dice'])) if val_metrics['dice'] else float('nan')
        val_sensitivity = float(np.mean(val_metrics['sensitivity'])) if val_metrics['sensitivity'] else float('nan')
        val_specificity = float(np.mean(val_metrics['specificity'])) if val_metrics['specificity'] else float('nan')
        val_precision = float(np.mean(val_metrics['precision'])) if val_metrics['precision'] else float('nan')

        if val_dice > best_val: 
            best_val = val_dice
            torch.save({
                'epoch': epoch, 
                'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': opt.state_dict(),
                'best_val': best_val
            }, MODEL_PATH)
            print(f"Saved best model at epoch {epoch} with Dice: {best_val:.4f}")

        # Calculate overall metrics from accumulated confusion matrix
        total_tp = val_metrics['tp']
        total_fp = val_metrics['fp']
        total_fn = val_metrics['fn']
        total_tn = val_metrics['tn']

        overall_sensitivity = total_tp / (total_tp + total_fn + 1e-8)
        overall_specificity = total_tn / (total_tn + total_fp + 1e-8)
        overall_precision = total_tp / (total_tp + total_fp + 1e-8)
        overall_dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + 1e-8)

        # scheduler.step(val_dice)

        print(f"Epoch: {epoch}/{EPOCHS} | train_loss: {avg_loss:.4f} | "
              f"val_dice_score: {val_dice:.4f} | val_sens: {val_sensitivity:.4f} | "
              f"val_spec: {val_specificity:.4f} | val_prec: {val_precision:.4f} | "
              f"time: {t1-t0:.1f}s")
        print(f"Overall - dice: {overall_dice:.4f} | sens: {overall_sensitivity:.4f} | "
              f"spec: {overall_specificity:.4f} | prec: {overall_precision:.4f}")
        # break

    print(f'Training finished')

    if os.path.exists(MODEL_PATH):
        print(f'Loading best model from {MODEL_PATH}')
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()
    else: 
        print(f'No trained model found!')
    # Testing
    print(f'Testing with strong mask/labels...')
    model.eval()
    # test_dices = []
    test_metrics = {
        'dice': [], 'sensitivity': [], 'specificity': [], 'precision': [], 
        'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0
    }
    with torch.no_grad():
        for img, s_mask, w_mask in test_loader:
            img = img.to(DEVICE)
            logits = model(img)
            probs = torch.sigmoid(logits).cpu().numpy()[:, 0]
            preds = (probs > 0.5).astype(np.uint8)
            mask_np = s_mask.numpy().astype(np.uint8)
            # Calculate metrics for this batch
            batch_metrics = calculate_metrics_batch(preds, mask_np)
            # Accumulate metrics
            for key in ['dice', 'sensitivity', 'specificity', 'precision']:
                test_metrics[key].append(batch_metrics[key])
            # Accumulate confusion matric components
            for key in ['tp', 'fp', 'fn', 'tn']:
                test_metrics[key] += batch_metrics[key]

    # Calculate average test metrics
    test_dice = float(np.mean(test_metrics['dice'])) if test_metrics['dice'] else float('nan')
    test_sensitivity = float(np.mean(test_metrics['sensitivity'])) if test_metrics['sensitivity'] else float('nan')
    test_specificity = float(np.mean(test_metrics['specificity'])) if test_metrics['specificity'] else float('nan')
    test_precision = float(np.mean(test_metrics['precision'])) if test_metrics['precision'] else float('nan')
    # Calculate overall metrics from accumulated confusion matrix
    total_tp = test_metrics['tp']
    total_fp = test_metrics['fp']
    total_fn = test_metrics['fn']
    total_tn = test_metrics['tn']
    
    test_overall_dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + 1e-8)
    test_overall_sensitivity = total_tp / (total_tp + total_fn + 1e-8)
    test_overall_specificity = total_tn / (total_tn + total_fp + 1e-8)
    test_overall_precision = total_tp / (total_tp + total_fp + 1e-8)

    print(f'\n📊 FINAL TEST RESULTS:')
    print(f'Average Metrics:')
    print(f'  Dice:      {test_dice:.4f}')
    print(f'  Sensitivity: {test_sensitivity:.4f}')
    print(f'  Specificity: {test_specificity:.4f}')
    print(f'  Precision:   {test_precision:.4f}')
    
    print(f'\nOverall Metrics (from confusion matrix):')
    print(f'  Dice:      {test_overall_dice:.4f}')
    print(f'  Sensitivity: {test_overall_sensitivity:.4f}')
    print(f'  Specificity: {test_overall_specificity:.4f}')
    print(f'  Precision:   {test_overall_precision:.4f}')
    
    print(f'\nConfusion Matrix Components:')
    print(f'  True Positives:  {total_tp}')
    print(f'  False Positives: {total_fp}')
    print(f'  False Negatives: {total_fn}')
    print(f'  True Negatives:  {total_tn}')

    print(f'Testing with weak mask/labels...')
    model.eval()
    # test_dices = []
    test_metrics = {
        'dice': [], 'sensitivity': [], 'specificity': [], 'precision': [], 
        'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0
    }
    with torch.no_grad():
        for img, s_mask, w_mask in test_loader:
            img = img.to(DEVICE)
            logits = model(img)
            probs = torch.sigmoid(logits).cpu().numpy()[:, 0]
            preds = (probs > 0.5).astype(np.uint8)
            mask_np = w_mask.numpy().astype(np.uint8)
            # Calculate metrics for this batch
            batch_metrics = calculate_metrics_batch(preds, mask_np)
            # Accumulate metrics
            for key in ['dice', 'sensitivity', 'specificity', 'precision']:
                test_metrics[key].append(batch_metrics[key])
            # Accumulate confusion matric components
            for key in ['tp', 'fp', 'fn', 'tn']:
                test_metrics[key] += batch_metrics[key]

    # Calculate average test metrics
    test_dice = float(np.mean(test_metrics['dice'])) if test_metrics['dice'] else float('nan')
    test_sensitivity = float(np.mean(test_metrics['sensitivity'])) if test_metrics['sensitivity'] else float('nan')
    test_specificity = float(np.mean(test_metrics['specificity'])) if test_metrics['specificity'] else float('nan')
    test_precision = float(np.mean(test_metrics['precision'])) if test_metrics['precision'] else float('nan')
    # Calculate overall metrics from accumulated confusion matrix
    total_tp = test_metrics['tp']
    total_fp = test_metrics['fp']
    total_fn = test_metrics['fn']
    total_tn = test_metrics['tn']
    
    test_overall_dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + 1e-8)
    test_overall_sensitivity = total_tp / (total_tp + total_fn + 1e-8)
    test_overall_specificity = total_tn / (total_tn + total_fp + 1e-8)
    test_overall_precision = total_tp / (total_tp + total_fp + 1e-8)

    print(f'\n📊 FINAL TEST RESULTS:')
    print(f'Average Metrics:')
    print(f'  Dice:      {test_dice:.4f}')
    print(f'  Sensitivity: {test_sensitivity:.4f}')
    print(f'  Specificity: {test_specificity:.4f}')
    print(f'  Precision:   {test_precision:.4f}')
    
    print(f'\nOverall Metrics (from confusion matrix):')
    print(f'  Dice:      {test_overall_dice:.4f}')
    print(f'  Sensitivity: {test_overall_sensitivity:.4f}')
    print(f'  Specificity: {test_overall_specificity:.4f}')
    print(f'  Precision:   {test_overall_precision:.4f}')
    
    print(f'\nConfusion Matrix Components:')
    print(f'  True Positives:  {total_tp}')
    print(f'  False Positives: {total_fp}')
    print(f'  False Negatives: {total_fn}')
    print(f'  True Negatives:  {total_tn}')


    
