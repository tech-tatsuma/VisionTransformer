import torch
import torchvision
import torch.optim as optim

import numpy as np
import random
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from models.vit import VisionTransformer

# setting seed
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train():
    # Set the random seed for reproducibility
    seed_everything(42)

    # Set up the device (GPU or CPU)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters setup
    epochs = 20
    learning_rate = 3e-4
    output_dir = "mnist_outputs/"

    # set the preprocess operations to be performed on train/val/test samples
    MNIST_preprocess = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5,), (0.5,))])
    # download MNIST training set and reserve 50000 for training
    train_dataset = torchvision.datasets.MNIST(root='data/MNIST/training', train=True, download=True, transform=MNIST_preprocess)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])

    # download MNIST test set
    test_set = torchvision.datasets.MNIST(root='data/MNIST/testing', train=False, download=True, transform=MNIST_preprocess)

    # define the data loaders using the datasets
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=32, shuffle=False)

    model = VisionTransformer(embed_dim=256,
                          hidden_dim=256*3,
                          num_heads=8,
                          num_layers=6,
                          patch_size=7,
                          num_channels=1,
                          num_patches=16,
                          num_classes=10,
                          dropout=0.2)

    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    model_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # set a scheduler to decay the learning rate by 0.1 on the 100th 150th epochs
    model_scheduler = optim.lr_scheduler.MultiStepLR(model_optimizer,milestones=[100, 150], gamma=0.1)

    val_loss_min = None
    val_loss_min_epoch = 0

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(epochs)):

        train_loss = 0.0
        val_loss = 0.0

        # Set model to training mode
        model.train()

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            # zero the parameter gradients
            model_optimizer.zero_grad()

            outputs = model(imgs)

            loss = loss_fn(outputs, labels)
            loss.backward()
            model_optimizer.step()

            train_loss += loss.item()

        # Calculate average training loss
        train_loss /= len(train_loader)
        # Record training loss
        train_losses.append(train_loss)

        # step the scheduler for the learning rate decay
        model_scheduler.step()

        model.eval()

        val_correct = 0  # Correct predictions count
        val_total = 0    # Total predictions count

        with torch.no_grad():
            for imgs, labels in val_loader:
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

    all_test_preds = []
    all_test_labels = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            all_test_preds.extend(predicted.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_test_labels, all_test_preds, labels=range(10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Test Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))

if __name__=="__main__":
    train()