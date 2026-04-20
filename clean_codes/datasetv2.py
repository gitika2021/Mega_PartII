import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class LightCurveDataset(Dataset):
    def __init__(self, folder: str, data_type: str,device: str):
        # self.lc_folder = os.path.join(folder, "LC10")
        # self.img_folder = os.path.join(folder, "OM10")
        self.lc_folder = os.path.join(folder)
        self.img_folder = os.path.join(folder)
        
        self.device=device
        self.data_type = data_type
        
        
        self.base_names = []
        lc_files_paths = sorted(glob.glob(os.path.join(self.lc_folder, f"{self.data_type}_*LC.npy")))
        print("lc_files_paths",lc_files_paths)
        for lc_path in lc_files_paths:
            base_name = os.path.basename(lc_path).replace('LC.npy', '')
            
            self.base_names.append(base_name)

        self.lc_paths = [os.path.join(self.lc_folder, f"{name}LC.npy") for name in self.base_names]
        #self.depth_paths = [os.path.join(self.lc_folder, f"{name}_meta.npy") for name in self.base_names]
        print("self.lc_files_paths",self.lc_files_paths)
        
        self.img_paths = [os.path.join(self.img_folder, f"{name.strip()}.npy") for name in self.base_names]
        print("self.img_paths",self.img_paths)
        
        for i, path in enumerate(self.img_paths):
            if not os.path.exists(path):
                no_space_path = os.path.join(self.img_folder, f"{self.base_names[i].strip()}.npy")
                if os.path.exists(no_space_path):
                    self.img_paths[i] = no_space_path
                else:
                    raise FileNotFoundError(f"Could not find image file for {self.base_names[i]}: {path} or {no_space_path}")

        self.lc_counts = [np.load(path, mmap_mode='r').shape[0] for path in self.lc_paths]
        self.cumulative_counts = np.cumsum(self.lc_counts)
        self.total_samples = self.cumulative_counts[-1] if self.cumulative_counts.size > 0 else 0
        self.lc_data = []
        self.depth_data = []
        self.img_data = []

        for lc_path, img_path in zip(self.lc_paths, self.img_paths):
            self.lc_data.append(np.load(lc_path))
            #self.depth_data.append(np.load(depth_path))
            self.img_data.append(np.load(img_path))

        self.lc_data =    torch.tensor(np.concatenate(self.lc_data), dtype=torch.float32).to(self.device)
        #self.depth_data = torch.tensor(np.concatenate(self.depth_data), dtype=torch.float32).to(self.device)
        #self.depth_data[:,-1]=1/self.depth_data[:,-1]
        self.img_data =   torch.tensor(np.concatenate(self.img_data), dtype=torch.float32).to(self.device)
        print(f'for lc max: {self.lc_data.max()} , min: {self.lc_data.min()}')
        #print(f'for metadata max: {self.depth_data.max()} , min: {self.depth_data.min()}')

    def __len__(self):
        return len(self.lc_data)

    def __getitem__(self, idx: int):
        lc_tensor = self.lc_data[idx].unsqueeze(0)
        #depth_tensor = self.depth_data[idx]
        img_tensor = self.img_data[idx].unsqueeze(0)
        return lc_tensor,0, img_tensor
