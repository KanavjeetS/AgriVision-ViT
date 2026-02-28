<div align="center">
  <img src="https://img.shields.io/badge/Domain-Computer%20Vision-blueviolet?style=for-the-badge" alt="Domain"/>
  <img src="https://img.shields.io/badge/Architecture-Swin%20Transformer-orange?style=for-the-badge" alt="Architecture"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch" alt="PyTorch"/>
</div>

<h1 align="center">🌾 AgriVision-ViT</h1>
<p align="center"><b>Hyperspectral Crop Disease Segmentation & Yield Regression via Swin Transformers.</b></p>
<hr>

## 🛰️ The Mission
Satellites and high-altitude drones capture far more than just RGB. They capture Near-Infrared (NIR) and Red-Edge spectral bands. **AgriVision-ViT** modifies a state-of-the-art **Swin Transformer V2** to accept 5-channel multi-spectral imagery.

By attaching a dual-head regressor and segmentation path outfitted with **Convolutional Block Attention Modules (CBAM)**, solving microscopic crop blight becomes computationally efficient and hyper-accurate.

## 🧠 SOTA Core Design

```mermaid
flowchart LR
    A[5-Channel Drone Image\n(RGB + NIR + RE)] --> B(Swin-V2 Tiny Backbone)
    B -->|Hidden Tensors| C{Feature Split}
    
    C --> D[Yield Regressor]
    D --> E((Tons / Hectare))
    
    C --> F[CBAM Attention Layer]
    F --> G[ConvTranspose2d Segmentation]
    G --> H((Blight Pixel Mask))
    
    style F fill:#ffcc00,stroke:#333
    style A fill:#ccffcc,stroke:#333
```

## 🔥 Hyper-Optimizations
*   **Swin-V2 Patch Modification**: The first 2D Convolution layer is programmatically altered to consume arbitrary channels `(in_channels=5)`.
*   **Dual-Objective Loss**: Blight segmentation is handled via CrossEntropy, while total yield output uses Huber Loss.
*   **CBAM (Convolutional Block Attention Module)**: Specifically injected into the segmentation upsampler. Attends to both spatial locations (where is the blight?) and channels (which spectral band matters most?) to isolate pathological crop decay.

## 🚀 Usage Guide

```bash
# 1. Prepare simulated drone captures
python fetch_dataset.py

# 2. View model topology and run training batch
python main.py
```

<br>
<div align="center">
  <i>Developed for precision agriculture routing at global scale. By @KanavjeetS.</i>
</div>
