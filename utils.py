import torch
import matplotlib.pyplot as plt
import numpy as np

def dice_loss_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    targets = targets.unsqueeze(1)
    num = 2.0 * (probs * targets).sum(dim=(2,3))
    den = probs.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + eps
    loss = 1.0 - (num / den)
    return loss.mean()

def dice_coef(y_true, y_pred, eps=1e-6):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    inter = (y_true * y_pred).sum()
    return (2*inter + eps) / (y_true.sum() + y_pred.sum() + eps)

def dice_score(y_true, y_pred):
    y_true = (y_true > 0.5).float().cpu()
    y_pred = (y_pred > 0.5).float().cpu()
    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred.view(-1)

    intersection = torch.sum(y_true_flat * y_pred_flat)
    sum_masks = torch.sum(y_true_flat) + torch.sum(y_pred_flat)
    # print(intersection, sum_masks)

    if sum_masks == 0: 
        return 1.0

    dice_score = (2. * intersection) / sum_masks
    return dice_score.item()

def visualize_results(gan_model, dataloader, num_samples=3):
    gan_model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            if isinstance(batch, (list, tuple)):
                images, masks = batch[0], batch[1]
            elif isinstance(batch, dict):
                images, masks = batch['image'], batch['mask']
            
            images = images.to(gan_model.device)
            masks = masks.to(gan_model.device)

            # Generate predictions
            fake_masks = torch.sigmoid(gan_model.generator(images))
            fake_masks_binary = (fake_masks > 0.5).float()

            # Plot
            img = images[0].cpu().permute(1, 2, 0)
            true_mask = masks[0].cpu().squeeze()
            pred_mask = fake_masks_binary[0].cpu().squeeze()

            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(true_mask, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred_mask, cmap='gray')
            axes[i, 2].set_title('Generated Mask')
            axes[i, 2].axis('off')

    plt.tight_layout()
    # plt.savefig()

def to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
        return data
    return data

def calculate_metrics(preds, targets):
    preds, targets = to_numpy(preds), to_numpy(targets)
    preds = preds.astype(bool)
    targets = targets.astype(bool)
    # Calculate confusion matrix
    tp = np.sum(preds & targets)
    fp = np.sum(preds & ~targets)
    fn = np.sum(~preds & targets)
    tn = np.sum(~preds & ~targets)

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)

    # Calculate dice score
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)

    return {
        'dice': dice, 
        'sensitivity': sensitivity, 
        'specificity': specificity, 
        'precision': precision, 
        'tp': tp, 
        'fp': fp, 
        'fn': fn, 
        'tn': tn
    }

def calculate_metrics_batch(preds, targets):
    batch_metrics = []
    for pred, target in zip(preds, targets):
        metrics = calculate_metrics(pred, target)
        batch_metrics.append(metrics)
    avg_metrics = {}
    for key in batch_metrics[0].keys():
        if key in ['tp', 'fp', 'fn', 'tn']:
            avg_metrics[key] = sum(m[key] for m in batch_metrics)
        else: 
            avg_metrics[key] = np.mean([m[key] for m in batch_metrics])

    return avg_metrics

