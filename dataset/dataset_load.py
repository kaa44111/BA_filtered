import sys
import os

# Initialisierung des PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.insert(0,project_path)

import random 

import torch
import torchvision
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm

from dataset.dataset import CustomDataset
from torchvision.transforms import v2

import wandb

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data parameters
num_classes = 10
input_shape = (1, 28, 28)


def load(dataset_name, val_split=0.2):
    """
    Load the dataset using CustomDataset class and split into train, validation, and test sets.
    
    Args:
        dataset_name (str): Name of the dataset directory.
        val_split (float): Fraction of the training set to use as validation.

    Returns:
        dict: Dictionary containing the datasets for 'train', 'val', and 'test'.
    """
    # Define directories
    train_dir = f"data/{dataset_name}"
    #train_dir = f"data/{dataset_name}/train"
    #test_dir = f"data/{dataset_name}/test"

    # Define transformations
    transform = v2.Compose([
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    mask_transform = v2.Compose([
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=False),
    ])

    # Load full training data using CustomDataset
    full_train_dataset = CustomDataset(root_dir=train_dir, dataset_name=dataset_name, transform=transform, mask_transform=mask_transform, is_labeled=True)

    # Calculate the number of samples for validation
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size

    # Split dataset into training and validation sets
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])

    # Load test data using CustomDataset
    #test_dataset = CustomDataset(root_dir=test_dir, dataset_name=dataset_name, transform=transform, mask_transform=mask_transform, is_labeled=False)

    return {
        'train': train_dataset,
        'val': val_dataset,
        #'test': test_dataset
    }

def load_and_log(dataset_name, val_split=0.2):
    """
    Load datasets using the CustomDataset class, split into train, val, and test,
    and log them as W&B artifacts.
    
    Args:
        dataset_name (str): Name of the dataset directory.
        val_split (float): Fraction of the training set to use as validation.
    """
    # Start a W&B run with a specific job type and project name
    with wandb.init(project="artifacts-example", job_type="load-data") as run:
        
        # Load datasets using the load function
        datasets = load(dataset_name, val_split)
        names = ["train", "val"]

        # Modify the dataset name to remove or replace invalid characters
        sanitized_name = dataset_name.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss").replace(" ", "_")

        # Create a new W&B artifact for the dataset
        raw_data = wandb.Artifact(
            name=f"{sanitized_name}_split", type="dataset",
            description=f"Split of the {dataset_name} dataset into train/val",
            metadata={"source": f"data/{dataset_name}",
                      "sizes": {name: len(dataset) for name, dataset in datasets.items()}}
        )

        # Save the train, validation, and test datasets as .pt files in the artifact
        for name, data in zip(names, datasets.values()):
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                x, y = zip(*[(x, y) for x, y in data])  # Extract tensors from the dataset
                x = torch.stack(x)
                y = torch.stack(y)
                torch.save((x, y), file)

        # Log the artifact to W&B
        run.log_artifact(raw_data)

# Example usage
if __name__ == "__main__":
    datasets = load("Dichtflächen_Cropped", val_split=0.2)
    print(f"Training set: {len(datasets['train'])} samples")
    print(f"Validation set: {len(datasets['val'])} samples")
    #print(f"Test set: {len(datasets['test'])} samples")
    load_and_log("Dichtflächen_Cropped", val_split=0.2)