import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

class AgriSpectralDataset(Dataset):
    """
    Dataset loader for AgriVision-ViT.
    Expects a CSV with 'image_path', 'mask_path' (blight segmentation target), 
    and 'yield_target' (regression value).
    """
    def __init__(self, csv_file, img_size=(256, 256)):
        self.data = pd.read_csv(csv_file)
        self.img_size = img_size
        
        # In a real scenario, multispectral transforms are custom. Here we use standard torchvision as a backbone.
        self.img_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(), # Standard 0-1 scale
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # For this stub, we assume standard RGB images. 
        # A true multispectral (e.g. 5-band TIFF) would load via rasterio or specialized libs.
        img = Image.open(row['image_path']).convert('RGB')
        mask = Image.open(row['mask_path']).convert('L') # Class labels
        
        img_tensor = self.img_transform(img)
        mask_tensor = self.mask_transform(mask).squeeze(0).long()
        
        yield_val = torch.tensor(row['yield_target'], dtype=torch.float32)

        return {
            "image": img_tensor,
            "mask": mask_tensor, # e.g. 0=Background, 1=Healthy, 2=Blight
            "yield": yield_val
        }

def train_agri_vision(model, train_loader, epochs=5, lr=1e-4):
    """
    Dual-objective training loop for Segmentation + Regression.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # We have two loss functions:
    # 1. Pixel-wise classification for blight mapping
    criterion_seg = nn.CrossEntropyLoss()
    # 2. Global regression for crop yield predicting
    criterion_yield = nn.HuberLoss() # More robust to yield outliers than MSE
    
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    model.train()
    
    for epoch in range(epochs):
        total_seg_loss = 0
        total_yield_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            yields = batch['yield']
            
            # Note: Depending on output shape, we unsqueeze yields for loss broadcasting
            yields = yields.unsqueeze(1).to(device) 
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred_masks, pred_yields = model(images)
                
                # Resize pred_masks back to match ground-truth spatial resolution if the backbone downsampled
                # (Assuming spatial match from our UNet head for now)
                
                loss_seg = criterion_seg(pred_masks, masks)
                loss_yield = criterion_yield(pred_yields, yields)
                
                # Combined objective (alpha weighting could be applied)
                loss = loss_seg + (0.5 * loss_yield)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_seg_loss += loss_seg.item()
            total_yield_loss += loss_yield.item()
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Seg Loss: {loss_seg.item():.4f} | Yield Loss: {loss_yield.item():.4f}")
                
        print(f"=== Epoch {epoch+1} Complete ===")
        
    return model

if __name__ == "__main__":
    print("[AgriVision-ViT] Dual-Objective Training Pipeline Ready.")
