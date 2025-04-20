import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import jaccard_score
from dataloader import get_dataloaders
import matplotlib.pyplot as plt  
import segmentation_models_pytorch as smp  # SMP U-Net is added 
import numpy as np


# current directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# sub directories
TRAIN_ROOT = os.path.join(BASE_DIR, "train_set")
TEST_ROOT = os.path.join(BASE_DIR, "test_set")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
VIS_DIR = os.path.join(BASE_DIR, "visualizations")

# directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

print("BASE_DIR:", BASE_DIR)
print("TRAIN_ROOT:", TRAIN_ROOT)
print("TEST_ROOT:", TEST_ROOT)
print("CHECKPOINT_DIR:", CHECKPOINT_DIR)
print("VIS_DIR:", VIS_DIR)


# Dataloaders
train_loader, val_loader, test_loader = get_dataloaders(TRAIN_ROOT, TEST_ROOT)

# Control visualization with a flag
VISUALIZE_RESULTS = True

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

# -------------------------------
# SMP U-Net (Pretrained Backbone)
# -------------------------------
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=10  # matches mask labels 0-9
).to(device)

# Optimizer & Loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

def train_one_epoch(model, loader):
    model.train()
    running_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        if masks.max() >= 10 or masks.min() < 0:
            print("Invalid label in training batch! min:", masks.min().item(), "max:", masks.max().item())
            continue

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(loader)

def validate(model, loader):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)

            if masks.max() >= 10 or masks.min() < 0:
                print("Invalid label in val batch! min:", masks.min().item(), "max:", masks.max().item())
                continue

            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    return val_loss / len(loader)


# Training loop with early stopping
best_val_loss = float('inf')
epochs_no_improve = 0
early_stop_patience = 10
num_epochs = 100  

for epoch in range(1, num_epochs + 1):
    train_loss = train_one_epoch(model, train_loader)
    val_loss = validate(model, val_loader)
    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
        print("Saved best model.")
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epoch(s).")

    if epochs_no_improve >= early_stop_patience:
        print(f"Early stopping triggered after {epoch} epochs.")
        break


# Load and evaluate best model
model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pth")))

# prediction and visualization
model.eval()
iou_scores = []

with torch.no_grad():
    for batch_idx, (images, masks) in enumerate(test_loader):
        images, masks = images.to(device), masks.to(device)

        if masks.max() >= 10 or masks.min() < 0:
            print("Invalid label in test batch! min:", masks.min().item(), "max:", masks.max().item())
            continue

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        for pred, target in zip(preds, masks):
            pred_np = pred.cpu().numpy().flatten()
            target_np = target.cpu().numpy().flatten()
            score = jaccard_score(target_np, pred_np, average='macro', labels=list(range(10)), zero_division=0)
            iou_scores.append(score)



        if VISUALIZE_RESULTS: 
            # Renk haritası
            unique_classes = torch.unique(masks[0]).cpu().numpy()
            num_classes = len(unique_classes)
            cmap = plt.colormaps.get_cmap("tab20").resampled(num_classes)
            color_map = {cls: cmap(i)[:3] for i, cls in enumerate(unique_classes)}
            background_class = min(unique_classes)
            color_map[background_class] = (1.0, 1.0, 1.0)  # white background

            def colorize_mask(mask_tensor):
                mask_np = mask_tensor.cpu().numpy()
                colored_mask = np.zeros((*mask_np.shape, 3), dtype=np.float32)
                for cls, color in color_map.items():
                    colored_mask[mask_np == cls] = color
                return colored_mask

            for i in range(images.size(0)):
                plt.figure(figsize=(12, 4))

                # Giriş Görseli
                plt.subplot(1, 3, 1)
                plt.imshow(images[i].cpu().permute(1, 2, 0))
                plt.title("Input Image")
                plt.axis("off")

                # Gerçek Maske
                plt.subplot(1, 3, 2)
                plt.imshow(colorize_mask(masks[i]))
                plt.title("Ground Truth")
                plt.axis("off")

                # Tahmin Maskesi
                plt.subplot(1, 3, 3)
                plt.imshow(colorize_mask(preds[i]))
                plt.title("Prediction")
                plt.axis("off")

                plt.tight_layout()
                save_path = os.path.join(VIS_DIR, f"prediction_{batch_idx}_{i}.png")
                plt.savefig(save_path)
                plt.close()
                
                
mean_iou = sum(iou_scores) / len(iou_scores)
print(f"Test Mean IoU: {mean_iou:.4f}")
