import torch
from torchvision import transforms
import torch.optim as optim
from torch import nn

import numpy as np
import random
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from models.vit import VisionTransformer
from imagenet.dataloaders.dataset import ImageNetDataset
from imagenet.lamb import Lamb

import sys

# setting seed
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(learning_rate):
    
    seed_everything(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    epochs = 300
    learning_rate = learning_rate
    output_root_dir = "imagenet_output"
    output_dir = os.path.join(output_root_dir, f"lr_{learning_rate}")

    # PyTorchの変換定義
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),  # モデルに応じてサイズ調整
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageNetDataset(split="train", transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=516, shuffle=True, num_workers=2, pin_memory=True)

    val_dataset = ImageNetDataset(split="validation", transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=516, shuffle=False, num_workers=2, pin_memory=True)

    test_dataset = ImageNetDataset(split="test", transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=516, shuffle=False, num_workers=2, pin_memory=True)

    model = VisionTransformer(embed_dim=768,
                          hidden_dim=768*4,
                          num_heads=8,
                          num_layers=12,
                          patch_size=16,
                          num_channels=3,
                          num_patches=196,
                          num_classes=1000,
                          dropout=0.1)

    # Parallelize model if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    model_optimizer = Lamb(model.parameters(), lr=learning_rate, weight_decay=0.01)

    warmup_epochs = 5
    total_epochs = epochs
    scheduler = optim.lr_scheduler.LambdaLR(
        model_optimizer,
        lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    )

    val_loss_min = None
    val_loss_min_epoch = 0

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(epochs)):

        train_loss = 0.0
        val_loss = 0.0

        # Set model to training mode
        model.train()
        scaler = torch.cuda.amp.GradScaler()

        for imgs, labels in tqdm(train_loader, desc="Training", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            # zero the parameter gradients
            model_optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(model_optimizer)
            scaler.update()

            train_loss += loss.item()

        # Calculate average training loss
        train_loss /= len(train_loader)
        # Record training loss
        train_losses.append(train_loss)

        # step the scheduler for the learning rate decay
        scheduler.step()

        model.eval()

        val_correct = 0  # Correct predictions count
        val_total = 0    # Total predictions count

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Validation", leave=False):
                imgs, labels = imgs.to(device), labels.to(device)

                outputs = model(imgs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1) 
                val_total += labels.size(0)      
                val_correct += (predicted == labels).sum().item() 

            val_loss /= len(val_loader)
            val_accuracy = val_correct / val_total  # Calculate validation accuracy
            val_losses.append(val_loss)  # Record validation loss

        print(f'Epoch {epoch+1}, Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
        sys.stdout.flush()

        if val_loss_min is None or val_loss < val_loss_min:
            model_save_name = os.path.join(output_dir, "best.pt")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            torch.save(model.state_dict(), model_save_name)

            val_loss_min = val_loss
            val_loss_min_epoch = epoch

    # Plot and save the learning curve
    plt.figure(figsize=(15, 5))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.title("Training and Validation Loss")
    graph_save_name = os.path.join(output_dir, "learning_curve")

    plt.savefig(graph_save_name)

    # Generate Confusion Matrix
    model.load_state_dict(torch.load(model_save_name))
    model.eval()

    # Validation Phase
    model.eval()  # 評価モードに切り替え

    test_correct = 0  # 正解数
    test_total = 0    # 総サンプル数

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)  # 最も確率の高いラベルを取得
            test_total += labels.size(0)  # バッチ内のサンプル数を追加
            test_correct += (predicted == labels).sum().item()  # 正解数を追加

    # Validation Accuracy の計算と標準出力
    test_accuracy = test_correct / test_total
    print(f'Test Accuracy: {test_accuracy:.4f}')

    return val_loss_min

if __name__=="__main__":
    lr_list = [0.0001, 0.00001]

    best_loss = None
    best_lr = None

    for lr in lr_list:
        print(f"learning rate: {lr}")
        sys.stdout.flush()
        val_loss_min = train(lr)
        if best_loss is None or val_loss_min < best_loss:
            best_loss = val_loss_min
            best_lr = lr

    print(f"best learning rate: {best_lr}, validation loss: {best_loss}")