import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image

def fetch_agri_data(subset_size=1000, output_dir="data"):
    """
    Downloads and prepares an Agricultural Vision dataset.
    Since raw High-Res Satellite/Drone imagery (TIFFs) are massive (GBs per file),
    we simulate the extraction of a Kaggle-style Sentinel-2 crop dataset 
    from IndiaAI Kosh into local RGB + Mask PNGs.
    """
    print(f"[START] Initiating Agri-Vision Dataset Fetcher ({subset_size} samples)...")
    
    img_dir = os.path.join(output_dir, "images")
    mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    out_path = os.path.join(output_dir, "agri_corpus.csv")
    
    data_records = []
    
    try:
        print("[DOWNLOADING] Simulating Satellite multispectral extraction...")
        # Since we can't physically download terabytes of raw satellite TIFFs here,
        # we generate localized high-variance dummy RGB chips and segmentation masks
        # to stand-in for the drone imagery (Healthy vs Blight vs Background).
        
        for i in range(min(subset_size, 50)): # Cap at 50 physical files for speed/storage in this stub
            img_file = os.path.join(img_dir, f"farm_patch_{i:04d}.png")
            mask_file = os.path.join(mask_dir, f"farm_mask_{i:04d}.png")
            
            # Generate dummy 256x256 RGB farm field (noise)
            dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            Image.fromarray(dummy_img).save(img_file)
            
            # Generate dummy integer classification mask (0, 1, or 2 for blight)
            dummy_mask = np.random.randint(0, 3, (256, 256), dtype=np.uint8)
            Image.fromarray(dummy_mask).save(mask_file)
            
            # Simulated crop yield metric (e.g., kilograms per hectare)
            yield_val = np.random.uniform(100.0, 5000.0)
            
            data_records.append({
                "image_path": img_file,
                "mask_path": mask_file,
                "yield_target": yield_val
            })
            
        df = pd.DataFrame(data_records)
        df.to_csv(out_path, index=False)
        
        print(f"[SUCCESS] Farm image patches extracted: {len(df)}")
        print(f"[SAVED] Tabular Ground Truth saved to: {out_path}")
        
    except Exception as e:
        print(f"[ERROR] Error fetching dataset: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch AgriVision Satellite Data")
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    fetch_agri_data(subset_size=args.samples)
