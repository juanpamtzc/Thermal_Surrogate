import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MultiPhysicsDataset(Dataset):
    def __init__(self, data_dir, stats=None):
        self.data_dir = data_dir
        self.stats = stats 
        
        # --- DEFINE THE EXPECTED TENSORS ---
        required_files = [
            "temperature.npy", "u_converged.npy", "u_inlet.npy", 
            "geom_mask.npy", "grid_x.npy", "grid_y.npy"
        ]
        
        # --- FILTER OUT SKIPPED CASES ---
        self.case_dirs = [
            os.path.join(data_dir, d) for d in sorted(os.listdir(data_dir)) 
            if os.path.isdir(os.path.join(data_dir, d))
            and all(os.path.exists(os.path.join(data_dir, d, f)) for f in required_files)
        ]
        
        if not self.case_dirs:
            raise RuntimeError(f"No valid simulation cases found in: {data_dir}")

    def __len__(self):
        return len(self.case_dirs)

    def __getitem__(self, idx):
        case_path = self.case_dirs[idx]
        
        # Load Inputs
        grid_x = np.load(os.path.join(case_path, "grid_x.npy"))      
        grid_y = np.load(os.path.join(case_path, "grid_y.npy"))      
        geom_mask = np.load(os.path.join(case_path, "geom_mask.npy")) 
        u_inlet = np.load(os.path.join(case_path, "u_inlet.npy"))     
        
        # Load Targets
        target_T = np.load(os.path.join(case_path, "temperature.npy")) 
        target_U = np.load(os.path.join(case_path, "u_converged.npy"))

        # --- STRICT NORMALIZATION ---
        if self.stats:
            u_inlet = (u_inlet - self.stats['ui_min']) / (self.stats['ui_max'] - self.stats['ui_min'] + 1e-8)
            grid_x = (grid_x - self.stats['x_min']) / (self.stats['x_max'] - self.stats['x_min'] + 1e-8)
            grid_y = (grid_y - self.stats['y_min']) / (self.stats['y_max'] - self.stats['y_min'] + 1e-8)
            
            target_T = (target_T - self.stats['t_min']) / (self.stats['t_max'] - self.stats['t_min'] + 1e-8)
            target_U = (target_U - self.stats['uc_min']) / (self.stats['uc_max'] - self.stats['uc_min'] + 1e-8)

        # Input Tensor X: [4, Height, Width]
        X = np.stack([geom_mask, u_inlet, grid_x, grid_y], axis=0).astype(np.float32)
        
        # Target Tensor Y: [2, Height, Width] -> Predicting both Temp and Velocity
        Y = np.stack([target_T, target_U], axis=0).astype(np.float32)

        return torch.from_numpy(X), torch.from_numpy(Y)

def get_dataloaders(train_dir, test_dir, batch_size=4, num_workers=4):
    print("📊 Computing global normalization bounds for Multi-Physics...")
    
    required_files = [
        "temperature.npy", "u_converged.npy", "u_inlet.npy", 
        "geom_mask.npy", "grid_x.npy", "grid_y.npy"
    ]
    
    # Filter cases to avoid crashing on skipped directories
    train_cases = [
        os.path.join(train_dir, d) for d in sorted(os.listdir(train_dir)) 
        if os.path.isdir(os.path.join(train_dir, d))
        and all(os.path.exists(os.path.join(train_dir, d, f)) for f in required_files)
    ]
    
    if not train_cases:
        raise RuntimeError(f"No valid training cases found in {train_dir} to compute stats!")
    
    t_mins, t_maxs, ui_mins, ui_maxs, uc_mins, uc_maxs = [], [], [], [], [], []
    x_mins, x_maxs, y_mins, y_maxs = [], [], [], []

    for case in train_cases:
        t_mins.append(np.min(np.load(os.path.join(case, "temperature.npy"))))
        t_maxs.append(np.max(np.load(os.path.join(case, "temperature.npy"))))
        ui_mins.append(np.min(np.load(os.path.join(case, "u_inlet.npy"))))
        ui_maxs.append(np.max(np.load(os.path.join(case, "u_inlet.npy"))))
        uc_mins.append(np.min(np.load(os.path.join(case, "u_converged.npy"))))
        uc_maxs.append(np.max(np.load(os.path.join(case, "u_converged.npy"))))
        x_mins.append(np.min(np.load(os.path.join(case, "grid_x.npy"))))
        x_maxs.append(np.max(np.load(os.path.join(case, "grid_x.npy"))))
        y_mins.append(np.min(np.load(os.path.join(case, "grid_y.npy"))))
        y_maxs.append(np.max(np.load(os.path.join(case, "grid_y.npy"))))

    stats = {
        't_min': np.min(t_mins), 't_max': np.max(t_maxs),
        'ui_min': np.min(ui_mins), 'ui_max': np.max(ui_maxs),
        'uc_min': np.min(uc_mins), 'uc_max': np.max(uc_maxs),
        'x_min': np.min(x_mins), 'x_max': np.max(x_maxs),
        'y_min': np.min(y_mins), 'y_max': np.max(y_maxs),
    }

    train_dataset = MultiPhysicsDataset(data_dir=train_dir, stats=stats)
    test_dataset = MultiPhysicsDataset(data_dir=test_dir, stats=stats)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader, stats