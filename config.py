import torch

config = {
    'epochs': 30,
    'batch_size': 25,
    'learning_rate': 1e-4,
    'dataset': 'Dichtflächen',
    'data_dir': 'data\data_modified\Dichtflächen_Cropped\patched_NIO',
    'model_name': 'UNetBatchNorm',
    'num_classes': 1,
}

def initialize_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")