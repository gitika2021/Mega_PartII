import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

def pt_compute_depths(light_curves):
    min_vals = torch.min(light_curves, dim=1)[0]
    depths = 1.0 - min_vals
    return depths

def pt_scale_vertically(lcs):
    min_vals = torch.min(lcs, dim=1, keepdim=True)[0]  # (N, 1)
    max_vals = torch.max(lcs, dim=1, keepdim=True)[0]  # (N, 1)
    range_vals = max_vals - min_vals
    range_vals = torch.where(range_vals == 0, torch.ones_like(range_vals), range_vals)
    scaled = (lcs - min_vals) / range_vals
    return scaled

def pt_extend_ltcrv(lcs, total_length=150):
    padded_curves = []
    for curve in tqdm(lcs):
        orig_len = curve.shape[0]
        pad_len = total_length - orig_len
        start = pad_len // 2
        end = pad_len - start        
        padded_curve = F.pad(curve, (start, end), 'constant', 1.0)
        padded_curves.append(padded_curve)
    return torch.stack(padded_curves)

def pt_scale_horizontally(lcs, indices, target_length=120):
    resized_curves = []
    device = lcs.device 
    for curve, idx in tqdm(zip(lcs, indices)):
        left, right, region_len = idx[0], idx[1], idx[2]
        if region_len >= 3 and left >= 0 and right <= curve.shape[0]:
            segment = curve[left:right]            
            segment_reshaped = segment.view(1, 1, -1)
            
            resized = F.interpolate(
                segment_reshaped, 
                size=target_length, 
                mode='linear', 
                align_corners=False
            )            
            resized_curves.append(resized.view(-1))
        else:
            resized_curves.append(torch.zeros(target_length, dtype=torch.float32, device=device))
            
    return torch.stack(resized_curves)

def pt_find_transit_regions(lcs, threshold=0.99):
    results = []
    S_full = lcs.shape[1]
    device = lcs.device
    
    for curve in tqdm(lcs):
        min_val = torch.min(curve)
        max_val = torch.max(curve)
        range_val = max_val - min_val        
        scaled = (curve - min_val) / (range_val + 1e-8)
        mask = (scaled < threshold).int()
        n = torch.sum(mask)
        if n >= 3:
            center = S_full // 2
            half_width = (n * 2) // 3
            left = torch.clamp(torch.tensor(center - half_width), min=0)
            right = torch.clamp(torch.tensor(center + half_width), max=S_full)
            region_length = right - left            
            results.append(torch.tensor([left, right, region_length], dtype=torch.int32))
        else:
            results.append(torch.tensor([-1, -1, 0], dtype=torch.int32))
    return torch.stack(results).to(device)

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            Na = str(sys.argv[1])

        try:
            #LCs_arr = np.load(f"LC10/{Na}.npy")
            LCs_arr = np.load(f"{Na}.npy")
            print("File found")
        except FileNotFoundError:
            print("File not found. Using placeholder data for demonstration.")
            LCs_arr = np.ones((10, 100))
            LCs_arr[:, 40:60] = np.linspace(1.0, 0.8, 20)
            LCs_arr[:, 60:80] = np.linspace(0.8, 1.0, 20)

        N, T = LCs_arr.shape
        print('LCs_arr.shape',LCs_arr.shape)

        lcs_tensor = torch.tensor(LCs_arr, dtype=torch.float32)
        ver_scaled = pt_scale_vertically(lcs_tensor)
        print('ver scaled')
        ver_scaled_np = ver_scaled.detach().cpu().numpy()


        np.save(f"{Na}_processed.npy",ver_scaled_np)
        
        plt.figure(figsize=(12, 6))
        for i in tqdm(range(10)):
            plt.plot(ver_scaled_np[i], label=f"LC {i}" if N < 15 else None) 
        plt.title("Final Preprocessed Light Curves (PyTorch)")
        plt.xlabel("Scaled Time")
        plt.ylabel("Flux (0-1 )")
        if N < 15:
            plt.legend()
        plt.grid(True)
        plt.savefig(f"LC10/{Na}.png")

        print(f"Processed tensor shape: {ver_scaled.shape}")
        print(f"Computed depths: {depths}")
    except:
        print('arguments?')
