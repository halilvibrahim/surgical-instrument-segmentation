import os
import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch

class MedicalSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, mask_size=(256, 256)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.mask_size = mask_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')  # Grayscale

        if self.transform:
            image = self.transform(image)

        # Resize mask
        mask = mask.resize(self.mask_size, resample=Image.NEAREST)
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask
    
def collect_image_mask_paths(root_dir):
    image_paths = []
    mask_paths = []

    video_dirs = sorted(glob.glob(os.path.join(root_dir, "video_*")))
    for video_dir in video_dirs:
        frames_dir = os.path.join(video_dir, "frames")
        masks_dir = os.path.join(video_dir, "segmentation")

        frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
        for frame_path in frame_files:
            filename = os.path.basename(frame_path)
            # Eğer dosya adı '_000016500.jpg' şeklindeyse, '_' karakterini kaldır
            if filename.startswith('_'):
                mask_filename = filename[1:].replace(".jpg", ".png")
            else:
                mask_filename = filename.replace(".jpg", ".png")
            mask_path = os.path.join(masks_dir, mask_filename)

            if os.path.exists(mask_path):
                image_paths.append(frame_path)
                mask_paths.append(mask_path)
            else:
                print(f"Mask not found for image: {frame_path}")

    return image_paths, mask_paths


def get_dataloaders(train_root, test_root, image_size=(256, 256), batch_size=32):
    # Eğitim ve doğrulama verileri
    train_image_paths, train_mask_paths = collect_image_mask_paths(train_root)

    # Geçersiz maskeleri filtrele
    valid_img_paths = []
    valid_mask_paths = []
    for img_path, mask_path in zip(train_image_paths, train_mask_paths):
        mask = Image.open(mask_path).convert('L').resize(image_size, resample=Image.NEAREST)
        mask_np = np.array(mask)
        if mask_np.max() <= 10 and mask_np.min() >= 0:
            valid_img_paths.append(img_path)
            valid_mask_paths.append(mask_path)
        else:
            print(f"Skipping mask: {mask_path} (min: {mask_np.min()}, max: {mask_np.max()})")

    # Eğitim ve doğrulama setlerine ayır
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        valid_img_paths, valid_mask_paths, test_size=0.2, random_state=42)

    # Test verileri
    test_image_paths, test_mask_paths = collect_image_mask_paths(test_root)

    # Geçersiz maskeleri filtrele
    valid_test_img_paths = []
    valid_test_mask_paths = []
    for img_path, mask_path in zip(test_image_paths, test_mask_paths):
        mask = Image.open(mask_path).convert('L').resize(image_size, resample=Image.NEAREST)
        mask_np = np.array(mask)
        if mask_np.max() <= 10 and mask_np.min() >= 0:
            valid_test_img_paths.append(img_path)
            valid_test_mask_paths.append(mask_path)
        else:
            print(f"Skipping test mask: {mask_path} (min: {mask_np.min()}, max: {mask_np.max()})")

    # Dönüşümler
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
    ])

    # Dataset ve DataLoader'lar
    train_dataset = MedicalSegmentationDataset(train_imgs, train_masks, transform, mask_size=image_size)
    val_dataset = MedicalSegmentationDataset(val_imgs, val_masks, transform, mask_size=image_size)
    test_dataset = MedicalSegmentationDataset(valid_test_img_paths, valid_test_mask_paths, transform, mask_size=image_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

