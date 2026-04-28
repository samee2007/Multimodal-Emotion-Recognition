import torch
from torch.utils.data import Dataset
import os

class MultimodalEmotionDataset(Dataset):
    def __init__(self, data_path="data/processed_real/dataset.pt"):
        """
        Loads the preprocessed MELD dataset features from disk.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} not found. Please run preprocessing scripts first.")
            
        self.data = torch.load(data_path, map_location=torch.device('cpu'))
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]
