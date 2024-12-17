import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

# データセットのロード
dataset = load_dataset('imagenet/dataloaders/imagenet-1k.py', trust_remote_code=True)

# PyTorchデータセット定義
class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, split, transform=None):
        self.data = dataset[split]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        image = example['image']
        label = example['label']

        # 画像がモノクロならRGBに変換
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
