import wandb
import os
from config import config, initialize_device
from dataset import get_dataloaders
from train import train_model
from UNetBatchNorm import UNetBatchNorm


def main():
    use_wandb = os.getenv('USE_WANDB', 'true').lower() in ('true', '1', 't')
    
    # Initialize W&B only if use_wandb is True
    if use_wandb:
        wandb.init(project="your_project_name", config=config)
    
    # Load the data
    dataloaders, dataset = get_dataloaders(root_dir=config['data_dir'], dataset_name=config['dataset'], batch_size=config['batch_size'])

    # Initialize the model
    model = UNetBatchNorm(config['num_classes']).to(initialize_device())

    # Train the model
    train_model(model, dataloaders, config, use_wandb)

    output_folder = os.path.join("test_results", config['model_name'])
    os.makedirs(output_folder, exist_ok=True)

if __name__ == "__main__":
    main()