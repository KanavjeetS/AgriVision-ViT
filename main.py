import os
import argparse
from torch.utils.data import DataLoader

# Import our custom architecture and training loop
from model import AgriVisionViT
from train import AgriSpectralDataset, train_agri_vision
from fetch_dataset import fetch_agri_data

def main():
    parser = argparse.ArgumentParser(description="AgriVision-ViT - Master Training Pipeline")
    parser.add_argument("--fetch", action="store_true", help="Download/Simulate the dataset first")
    parser.add_argument("--samples", type=int, default=100, help="Number of dataset rows to fetch/train on")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DataLoader")
    args = parser.parse_args()

    # 1. Pipeline: Data Ingestion
    data_dir = "data"
    csv_path = os.path.join(data_dir, "agri_corpus.csv")
    
    if args.fetch or not os.path.exists(csv_path):
        print("\n--- PHASE 1: DATA INGESTION ---")
        fetch_agri_data(subset_size=args.samples, output_dir=data_dir)
    else:
        print(f"\n--- PHASE 1: DATA INGESTION ---\n[INFO] Found existing dataset at {csv_path}")

    # 2. Pipeline: Dataset Loading
    print("\n--- PHASE 2: INITIALIZING DATALOADER ---")
    
    try:
        # Assuming our simulated images are standard RGB for now
        dataset = AgriSpectralDataset(csv_file=csv_path, img_size=(256, 256))
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        print(f"[SUCCESS] DataLoader initialized with {len(dataset)} samples across {len(train_loader)} batches.")
    except Exception as e:
        print(f"[ERROR] Error initializing DataLoader: {e}")
        return

    # 3. Pipeline: Model Initialization
    print("\n--- PHASE 3: MODEL ARCHITECTURE ---")
    print("Building Swin-Transformer (ViT) with Dual-Heads (Segmentation + Regression)...")
    try:
        # 3 Channels for RGB, predicting 3 classes (e.g., Background, Healthy, Blighted)
        model = AgriVisionViT(in_channels=3, num_classes=3)
        print("[SUCCESS] Swin Vision Transformer compiled.")
    except Exception as e:
        print(f"[ERROR] Error initializing Model: {e}")
        return

    # 4. Pipeline: Training Loop
    print("\n--- PHASE 4: PyTorch AMP TRAINING ---")
    print(f"Beginning Dual-Objective validation training for {args.epochs} epochs...\n")
    try:
        trained_model = train_agri_vision(model, train_loader, epochs=args.epochs)
        print("\n[SUCCESS] Training Pipeline Executed Successfully!")
    except Exception as e:
        print(f"\n[ERROR] Pipeline Halted during training: {e}")

if __name__ == "__main__":
    main()
