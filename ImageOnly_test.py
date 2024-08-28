import sys
import os

# Initialisierung des PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)
    
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
from torchvision.transforms import v2
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from utils.data_utils import compute_mean_std

matplotlib.use('TkAgg')  # Backend auf TkAgg umstellen


class ImageOnlyDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(self.image_dir), key=lambda x: int(''.join(filter(str.isdigit, x))))

        print(f"Found {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            img_name = os.path.join(self.image_dir, self.image_files[idx])
            image = Image.open(img_name).convert('RGB')
            image = v2.functional.pil_to_tensor(image).float() / 255.0

            if self.transform:
                image = self.transform(image)

            return image

        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            return None
        

def show_predictions(images, preds, idx):
    """
    Zeigt die Originalbilder und die vorhergesagten Masken für eine gegebene Index an.
    """
    fig, axes = plt.subplots(len(images), 2, figsize=(10, 5 * len(images)))

    for i in range(len(images)):
        # Original image
        img = images[idx + i].cpu().numpy().transpose((1, 2, 0))
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        # Predicted mask
        pred = preds[idx + i].cpu().squeeze().numpy()
        sns.heatmap(pred, ax=axes[i, 1], cmap='viridis')
        axes[i, 1].set_title('Predicted Mask')
        axes[i, 1].axis('off')

    plt.tight_layout()

    # # Anpassen der Fenstergröße und -position
    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())  # Fenster maximieren
    # # Alternativ: mng.window.setGeometry(x, y, width, height)
    # # Beispiel: mng.window.setGeometry(100, 100, 1280, 720)

    # # Anpassen der Fenstergröße und -position
    # mng = plt.get_current_fig_manager()
    # mng.window.wm_geometry("+100+100")  # Fensterposition setzen
    # mng.resize(1280, 720)  # Fenstergröße setzen

    plt.show()


def test(UNet,test_dir,trained_path):
    num_class = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = UNet(num_class).to(device)
    model.load_state_dict(torch.load(trained_path, map_location=device))
    model.eval()
    
    #mean, std = compute_mean_std('data_modified/RetinaVessel/train/grabs')

    transformations = v2.Compose([
        v2.RandomEqualize(p=1.0),
        v2.ToPureTensor(),
        v2.ToDtype(torch.float32, scale=True),
        #v2.Normalize(mean=mean, std=std)
    ])

    test_dataset = ImageOnlyDataset(test_dir, transform=transformations)
    test_loader = DataLoader(test_dataset, batch_size=3, shuffle=True, num_workers=0)

    images = next(iter(test_loader))
    images = images.to(device)

    pred = model(images)
    pred = F.sigmoid(pred)
    pred = pred.data.cpu()

    print(pred.shape)
    print(images.shape)

    # Show predictions
    show_predictions(images, pred, 0)

# if __name__ == '__main__':
#     try:
#         from models.UNet import UNet
#         #from models.UNetBatchNorm import UNetBatchNorm
#         #from models.UNetMaxPool import UNetMaxPool

#         test_dir = 'data_modified/RetinaVessel/test'
#         trained_path = 'train/results/RetinaVessel/test_train.pth'
#         test(UNet,test_dir,trained_path) 
               
#         # trained_path = 'UNetBatchNorm_RetinaVessel.pth'
#         # test(UNetBatchNorm,trained_path)

#     except Exception as e:
#         print(f"An error occurred: {e}")