import sys
import os

# Initialisierung des PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.utils
import time
import copy
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from metrics import calculate_metrics, calc_loss
import wandb
import time  # Import time module to track execution time
import matplotlib.pyplot as plt  # Import for plotting

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    print("{}: {}".format(phase, ", ".join(outputs)))

def train_model(model, dataloaders, config, use_wandb):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    
    best_loss = float('inf')
    total_training_time = 0  # Track total training time

    # Lists to store metrics for plotting
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(config['epochs']):
        print(f"Epoch {epoch+1}/{config['epochs']}")
        
        epoch_start_time = time.time()  # Start time of the epoch
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_samples += inputs.size(0)
                calculate_metrics(outputs, labels, metrics)  # Calculate metrics after each batch

            epoch_loss = metrics['loss'] / epoch_samples
            epoch_accuracy = metrics['pixel_accuracy'] / epoch_samples  # Assuming pixel accuracy is calculated

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_accuracy)
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_accuracy)

            print_metrics(metrics, epoch_samples, phase)  # Print the metrics at the end of each phase

            # Conditionally log metrics to W&B
            if use_wandb:
                wandb.log({f"{phase}_loss": epoch_loss, f"{phase}_accuracy": epoch_accuracy})
                for key, value in metrics.items():
                    wandb.log({f"{phase}_{key}": value / epoch_samples})

            # Save the best model
            if phase == 'val' and epoch_loss < best_loss:
                print("Saving best model...")
                best_loss = epoch_loss
                torch.save(model.state_dict(), f"{config['model_name']}_best.pth")

        scheduler.step()

    # Log total training time
    total_training_time = time.time() - epoch_start_time
    print(f"Total Training Time: {total_training_time:.2f} seconds")
    if use_wandb:
        wandb.log({"total_training_time": total_training_time})

    # Save the final model
    torch.save(model.state_dict(), f"{config['model_name']}.pth")

    # Plotting losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{config['model_name']}_loss.png")
    plt.show()

    # Plotting accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{config['model_name']}_accuracy.png")
    plt.show()