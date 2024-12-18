import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from utils.img2attn import img2attentionscores
from models.vit import VisionTransformer

image_size = 28
embed_dim=256
hidden_dim=embed_dim*3
num_heads=8
num_layers=6
patch_size=7
num_patches=16
num_channels=1
num_classes=10

def imgpath2heatmap(img_path, model_path, output_path):
    # Load and preprocess the image
    img = Image.open(img_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.load_state_dict(torch.load(model_path))
    img_tensor = img_tensor.squeeze(0)

    heat_map = img2attentionscores(img_tensor.unsqueeze(0), model, device, image_size, patch_size, num_heads, num_patches)

    # Convert image and heatmap to numpy arrays
    img_np = np.asarray(img_tensor.cpu())
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
    image_path = ""
    model_path = ""
    output_path = "output.png"
    imgpath2heatmap(image_path, model_path, output_path)