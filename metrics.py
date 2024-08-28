import sys
import os

# Initialisierung des PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)

import torch
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()

def calc_loss(pred, target, metrics, bce_weight=0.5):
    # Verwende Binary Cross Entropy (BCE) Loss
    bce = F.binary_cross_entropy_with_logits(pred, target)

    # Wende Sigmoid an, um die Ausgaben in den Bereich [0,1] zu skalieren
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    # Kombiniere BCE und Dice Loss
    loss = bce * bce_weight + dice * (1 - bce_weight)

    # Metriken für Monitoring aktualisieren
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def calculate_metrics(pred, target, metrics):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    metrics['mean_iou'] += mean_iou(pred, target).item()
    metrics['pixel_accuracy'] += pixel_accuracy(pred, target).item()
    metrics['mean_pixel_accuracy'] += mean_pixel_accuracy(pred, target).item()

def mean_iou(pred, target, eps=1e-6):
    # Convert predictions and targets to binary (0 or 1)
    pred = (pred > 0.5).int()
    target = target.int()

    intersection = (pred & target).float().sum((1, 2))
    union = (pred | target).float().sum((1, 2))
    iou = (intersection + eps) / (union + eps)
    return iou.mean()

def pixel_accuracy(pred, target):
    correct = (pred == target).float().sum()
    total = target.numel()
    return correct / total

def mean_pixel_accuracy(pred, target):
    class_accuracies = []
    for c in range(pred.size(1)):
        correct = (pred[:, c] == target[:, c]).float().sum()
        total = target[:, c].numel()
        class_accuracies.append(correct / total)
    return torch.mean(torch.tensor(class_accuracies))