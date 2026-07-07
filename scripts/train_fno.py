import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Ensure Python can find your 'src' directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_dataloaders
from src.model import ThermalFNO

def train():
    # --- 1. HARDWARE OPTIMIZATION ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- 2. HYPERPARAMETERS & PATHS ---
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "train")
    TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    EPOCHS = 100
    BATCH_SIZE = 16  
    LEARNING_RATE = 1e-3
    
    train_loader, test_loader = get_dataloaders(
        TRAIN_DIR, TEST_DIR, batch_size=BATCH_SIZE, num_workers=8
    )

    # --- 3. MODEL & OPTIMIZER ---
    model = ThermalFNO(modes1=12, modes2=32, width=32).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    # --- 4. TRAINING LOOP (Standard FP32) ---
    print("\n🔥 Starting FNO Training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        for X_batch, Y_batch in pbar:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            
            optimizer.zero_grad()
            
            # Standard Forward & Backward Pass
            predictions = model(X_batch)
            loss = criterion(predictions, Y_batch)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})
            
        train_loss /= len(train_loader.dataset)
        
        # --- 5. VALIDATION LOOP ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                
                predictions = model(X_batch)
                loss = criterion(predictions, Y_batch)
                    
                val_loss += loss.item() * X_batch.size(0)
                
        val_loss /= len(test_loader.dataset)
        scheduler.step()
        
        print(f"Epoch {epoch} Summary -> Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Save the best model weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(MODEL_DIR, "fno_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"🌟 New best model saved to {save_path}")

    print("\n✅ Training Complete!")

if __name__ == "__main__":
    train()