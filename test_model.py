import sys
import os

# Initialisierung des PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
from UNetBatchNorm import UNetBatchNorm
from dataset import get_dataloaders
from config import config

def show_predictions_with_heatmaps(dataloader, output_folder, num_images=5):    
    num_class = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = UNetBatchNorm(num_class).to(device)
    model.load_state_dict(torch.load('UNetBatchNorm_best.pth', map_location=device))
    model.eval()

    images, masks, preds = [], [], []

    for idx, (inputs, labels) in enumerate(dataloader):
        if idx >= num_images:
            break
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            pred = torch.sigmoid(outputs)

        images.extend(inputs.cpu().numpy())
        masks.extend(labels.cpu().numpy())
        preds.extend(pred.cpu().numpy())

    for i in range(len(images)):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original Image
        img = images[i].transpose(1, 2, 0)
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Ground Truth Mask
        mask = masks[i][0]
        sns.heatmap(mask, ax=axes[1], cmap='viridis')
        axes[1].set_title('Original Mask')
        axes[1].axis('off')

        # Predicted Mask
        pred = preds[i][0]
        sns.heatmap(pred, ax=axes[2], cmap='viridis', vmin=0.0, vmax=1.0)
        axes[2].set_title('Predicted Mask')
        axes[2].axis('off')

        plt.tight_layout()
        save_path = os.path.join(output_folder, f"heatmap_{i}.png")
        plt.savefig(save_path)
        plt.show()
        plt.close()

dataloaders, dataset = get_dataloaders(root_dir=config['data_dir'], dataset_name=config['dataset'], batch_size=config['batch_size'])

show_predictions_with_heatmaps(dataloader=dataloaders['val'],output_folder='test_results')