import torch
from torch import nn
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from utils.img2attn import img2attentionscores
from models.vit import VisionTransformer

image_size = 224
embed_dim=768
hidden_dim=768*4
num_heads=8
num_layers=12
patch_size=16
num_patches=196
num_channels=3
num_classes=1000

def imgpath2heatmap(img_path, model_path, output_path):
    # Load and preprocess the image
    img = Image.open(img_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img)
    model = VisionTransformer(embed_dim=embed_dim,
                          hidden_dim=hidden_dim,
                          num_heads=num_heads,
                          num_layers=num_layers,
                          patch_size=patch_size,
                          num_channels=num_channels,
                          num_patches=num_patches,
                          num_classes=num_classes)

    model = nn.DataParallel(model)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.load_state_dict(torch.load(model_path, weights_only=False))
    img_tensor = img_tensor.squeeze(0).squeeze(0)

    heat_map = img2attentionscores(img_tensor, model.module, device, image_size, patch_size, num_heads, num_patches)

    # Convert image and heatmap to numpy arrays
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * std + mean).clip(0, 1)
    heat_map_np = heat_map.detach().cpu().numpy()

    # Overlay the heatmap on the original image
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_np, cmap='gray')
    ax.imshow(heat_map_np, cmap='jet', alpha=0.5)  # Apply transparency to the heatmap
    ax.set_title('Overlayed Attention Map')
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labeltop=False, labelleft=False, labelright=False)

    plt.savefig(output_path)

if __name__=="__main__":
    image_path = "/home/furuya/VisionTransformer/image copy.png"
    model_path = "/home/furuya/VisionTransformer/VisionTransformer/imagenet_output/lr_0.0001/best.pt"
    output_path = "output.png"
    imgpath2heatmap(image_path, model_path, output_path)