import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_transformer

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x_cat))

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class AgriVisionViT(nn.Module):
    """
    Swin Transformer backbone adapted for Multi-Spectral Drone/Satellite Imagery.
    Provides segmentation masks for blight detection and dense regression for crop yield.
    """
    def __init__(self, in_channels=5, num_classes=10):
        super(AgriVisionViT, self).__init__()
        
        # We start with a base SwinV2-Tiny
        # We need to adapt the patch embedding for multispectral input (e.g., RGB + NIR + RE)
        self.swin = swin_transformer.swin_v2_t(weights=None)
        
        # Modify the first layer to accept arbitrary input channels
        original_conv = self.swin.features[0][0]
        self.swin.features[0][0] = nn.Conv2d(
            in_channels, 
            original_conv.out_channels, 
            kernel_size=original_conv.kernel_size, 
            stride=original_conv.stride, 
            padding=original_conv.padding
        )
        
        # Output heads
        hidden_dim = self.swin.head.in_features
        self.swin.head = nn.Identity() # Remove default classification head
        
        # Segmentation path (Blight detection) with injected CBAM attention
        self.segmentation_head = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 256, kernel_size=4, stride=4),
            nn.ReLU(),
            CBAM(256), # Hyper-Optimization: Emphasizing structural blight differences
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
        # Regression path (Yield prediction)
        self.yield_regressor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x is (B, C, 256, 256)
        features = self.swin(x) # (B, 768)
        
        # Spatial features for segmentation
        b, c = features.shape
        # (Assuming global pooled output from torchvision Swin, we dummy-expand for illustrative segmentation)
        # Real implementation would hook intermediate layers.
        spatial_feat = features.view(b, c, 1, 1).expand(-1, -1, 32, 32)
        
        seg_mask_small = self.segmentation_head(spatial_feat) # (B, num_classes, 128, 128)
        
        # Upsample to match the raw input image resolution (256x256)
        seg_mask = F.interpolate(seg_mask_small, size=(256, 256), mode='bilinear', align_corners=False)
        
        predicted_yield = self.yield_regressor(features)
        
        return seg_mask, predicted_yield

if __name__ == "__main__":
    print("[AgriVision-ViT] Initializing Multispectral Tensor (B, 5, 256, 256)...")
    model = AgriVisionViT(in_channels=5, num_classes=3)
    dummy_input = torch.randn(2, 5, 256, 256)
    seg, yield_pred = model(dummy_input)
    
    print(f"Segmentation Output Shape: {seg.shape}")
    print(f"Crop Yield Output Shape: {yield_pred.shape}")
    print("[SUCCESS] AgriVision-ViT built successfully.")
