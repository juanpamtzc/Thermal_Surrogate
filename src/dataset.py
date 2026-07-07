import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ThermalDataset(Dataset):
    """
    Custom PyTorch Dataset for loading and normalizing 2D OpenFOAM thermal simulation fields.
    """
    def __init__(self, data_dir, stats=None):
        self.data_dir = data_dir
        self.stats = stats  # Dictionary holding the Min/Max bounds for scaling
        
        # Robustly find all simulation directories
        self.case_dirs = [
            os.path.join(data_dir, d) for d in sorted(os.listdir(data_dir)) 
            if os.path.isdir(os.path.join(data_dir, d))
        ]
        
        if not self.case_dirs:
            raise RuntimeError(f"No simulation cases found in: {data_dir}")

    def __len__(self):
        return len(self.case_dirs)

    def __getitem__(self, idx):
        case_path = self.case_dirs[idx]
        
        # Load the extracted NumPy arrays
        grid_x = np.load(os.path.join(case_path, "grid_x.npy"))      
        grid_y = np.load(os.path.join(case_path, "grid_y.npy"))      
        geom_mask = np.load(os.path.join(case_path, "geom_mask.npy")) 
        u_inlet = np.load(os.path.join(case_path, "u_inlet.npy"))     
        target_T = np.load(os.path.join(case_path, "temperature.npy")) 

        # --- MIN-MAX NORMALIZATION ---
        if self.stats:
            # Mask is already 0 or 1, so no scaling needed
            # Adding 1e-8 to the denominator prevents DivisionByZero errors
            u_inlet = (u_inlet - self.stats['u_min']) / (self.stats['u_max'] - self.stats['u_min'] + 1e-8)
            grid_x = (grid_x - self.stats['x_min']) / (self.stats['x_max'] - self.stats['x_min'] + 1e-8)
            grid_y = (grid_y - self.stats['y_min']) / (self.stats['y_max'] - self.stats['y_min'] + 1e-8)
            target_T = (target_T - self.stats['t_min']) / (self.stats['t_max'] - self.stats['t_min'] + 1e-8)

        # Construct Input Tensor X: [Channels, Height, Width]
        input_channels = [
            geom_mask.astype(np.float32),
            u_inlet.astype(np.float32),
            grid_x.astype(np.float32),
            grid_y.astype(np.float32)
        ]
        X = np.stack(input_channels, axis=0)
        
        # Construct Target Tensor Y: [Channels, Height, Width]
        Y = np.expand_dims(target_T.astype(np.float32), axis=0)

        return torch.from_numpy(X), torch.from_numpy(Y)

def get_dataloaders(train_dir, test_dir, batch_size=4, num_workers=4):
    """Instantiates DataLoaders for training and testing, computing scaling bounds dynamically."""
    
    print("📊 Scanning training set to compute normalization bounds...")
    train_cases = [os.path.join(train_dir, d) for d in sorted(os.listdir(train_dir)) if os.path.isdir(os.path.join(train_dir, d))]
    
    t_mins, t_maxs = [], []
    u_mins, u_maxs = [], []
    x_mins, x_maxs = [], []
    y_mins, y_maxs = [], []

    # Read the dataset once to find the absolute maximums and minimums
    for case in train_cases:
        t_mins.append(np.min(np.load(os.path.join(case, "temperature.npy"))))
        t_maxs.append(np.max(np.load(os.path.join(case, "temperature.npy"))))
        u_mins.append(np.min(np.load(os.path.join(case, "u_inlet.npy"))))
        u_maxs.append(np.max(np.load(os.path.join(case, "u_inlet.npy"))))
        x_mins.append(np.min(np.load(os.path.join(case, "grid_x.npy"))))
        x_maxs.append(np.max(np.load(os.path.join(case, "grid_x.npy"))))
        y_mins.append(np.min(np.load(os.path.join(case, "grid_y.npy"))))
        y_maxs.append(np.max(np.load(os.path.join(case, "grid_y.npy"))))

    # Compile the global statistics dict
    stats = {
        't_min': np.min(t_mins), 't_max': np.max(t_maxs),
        'u_min': np.min(u_mins), 'u_max': np.max(u_maxs),
        'x_min': np.min(x_mins), 'x_max': np.max(x_maxs),
        'y_min': np.min(y_mins), 'y_max': np.max(y_maxs),
    }
    
    print(f"   --> Found Temperature Range: {stats['t_min']:.2f} to {stats['t_max']:.2f}")
    print(f"   --> Normalizing all data to [0, 1] range!")

    # Instantiate Datasets passing the SAME stats to both Train and Test
    train_dataset = ThermalDataset(data_dir=train_dir, stats=stats)
    test_dataset = ThermalDataset(data_dir=test_dir, stats=stats)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    return train_loader, test_loader