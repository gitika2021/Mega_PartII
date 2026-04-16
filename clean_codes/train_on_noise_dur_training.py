import sys
sys.path.append("/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/pyscripts")
sys.path.append("/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git")
sys.path.append("/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Python-Scripts")

import numpy as np
import matplotlib.pyplot as plt
from kepler_noise_sampler import *
import time
from lightkurve import LightCurve, LightCurveCollection, read, search_lightcurve
from pathlib import Path
import os 

import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import DataLoader
import numpy as np
from dataset import *
#from models2 import *
from models import *
import cProfile
import pstats
from utils import *
import matplotlib.pyplot as plt

# -------- GLOBALS (set inside workers) --------
kepler_lcs_error = None
median_error = None
bins = None


# -------- INITIALIZER (runs once per worker) --------
def init_worker(kepler_file):
    global kepler_lcs_error, median_error, bins

    kepler_lcs_error = np.load(kepler_file, mmap_mode='r')
    median_error = np.sqrt(np.median(kepler_lcs_error**2, axis=1))
    bins = create_noise_bins_Kepler(kepler_lcs_error, n=30)
    
# In main.py
#criterion = nn.MSELoss()
criterionb = symmetry_aware_bce

criterion = symmetry_aware_dice_loss
rng = np.random.default_rng(seed=42) # Optional: provide a seed for reproducibility
def train_gan(generator, traindataloader, valdataloader, snr, num_epochs=50, device="cuda",modelpath='Linear',n=1):
    optimizer_G = optim.AdamW(generator.parameters(), lr=1e-3, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, 'min', patience=15, factor=0.5)
    best_val_loss = float("inf") # Simplified this
    val_loss_counter = 0

    for epoch in range(num_epochs):
        generator.train()
        gbce_total = 0
        gmse_total = 0
        if epoch>100:
            snru=50+450*rng.random()
        elif epoch>60:
            snru=250+250*rng.random()
        elif epoch>30:
            snru=350+150*rng.random()
        else:
            snru=400+100*rng.random()
        for i, (lc_batch, real_depths, real_imgs) in enumerate(traindataloader):
            noisy_lc = add_noise_to_batch(lc_batch.squeeze(1), snru)
            #print(noisy_lc.shape)
            optimizer_G.zero_grad()
            gen_imgs = generator(noisy_lc.view(noisy_lc.shape[0],1,120))

            # lc_input = lc_batch.squeeze(1)
            # optimizer_G.zero_grad()
            # gen_imgs = generator(lc_input.view(lc_input.shape[0], 1, 120))
            
            g_loss_bce = symmetry_aware_bce(real_imgs.squeeze(),gen_imgs.squeeze()) 
            
            g_loss_mse = symmetry_aware_mse(real_imgs.squeeze(),gen_imgs.squeeze())
            if epoch>60:
                genlos = g_loss_bce
            else:
                genlos = g_loss_bce
            
            gmse_total += g_loss_mse.detach()
            gbce_total += g_loss_bce.detach()
            
            genlos.backward()
            if i == 1 and epoch%10==0:
                print_grad_stats(generator, i)
            optimizer_G.step()
            if i == 0 and epoch % 10 == 0:
                

                prob_map = gen_imgs[0].squeeze().detach().cpu().numpy()
                true_img = real_imgs[0].squeeze().detach().cpu().numpy()

                hard_shape = (prob_map > 0.5).astype(float)

                plt.figure(figsize=(12, 4))

                plt.subplot(1, 3, 1)
                plt.title("Ground Truth")
                plt.imshow(true_img, vmin=0, vmax=1, cmap='inferno')
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.title(f"Sigmoid Output (Confidence)\nMSE: {g_loss_mse.item():.4f}")
                plt.imshow(prob_map, vmin=0, vmax=1, cmap='inferno')
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.title("Thresholded > 0.5\n(Final Prediction)")
                plt.imshow(hard_shape, vmin=0, vmax=1, cmap='inferno')
                plt.axis('off')

                #plt.savefig(f"plots/debug_sigmoid_epoch_{epoch}.png")
                plt.savefig(f"/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/plots/debug_sigmoid_epoch_{epoch}.png")
                plt.close()

        avg_train_bce = gbce_total / len(traindataloader)
        avg_train_mse = gmse_total / len(traindataloader)
        
        val_bce_losses = []
        val_mse_losses = []
        val_total_losses = []
        
        generator.eval() 
        with torch.no_grad():
            for lc_batch, real_depths, real_imgs in valdataloader:
                noisy_lc = add_noise_to_batch(lc_batch.squeeze(1), snru)
                gen_imgs = generator(noisy_lc.view(noisy_lc.shape[0],1,120))

                # lc_input = lc_batch.squeeze(1)
                # gen_imgs = generator(lc_input.view(lc_input.shape[0], 1, 120))
            
                val_bce = symmetry_aware_bce(real_imgs.squeeze(), gen_imgs.squeeze())
                val_mse = symmetry_aware_mse(real_imgs.squeeze(), gen_imgs.squeeze())
                
                val_bce_losses.append(val_bce.item())
                val_mse_losses.append(val_mse.item())
                if epoch>60:
                    val_total_losses.append(val_mse.item()+val_bce.item())
                else:
                    val_total_losses.append(val_bce.item()+val_mse.item())
        
        avg_val_bce = np.mean(val_bce_losses)
        avg_val_mse = np.mean(val_mse_losses)
        avg_val_loss = np.mean(val_total_losses) 
        old_lr = optimizer_G.param_groups[0]['lr'] 
        
        scheduler.step(avg_val_loss)
        
        new_lr = optimizer_G.param_groups[0]['lr']        
        if new_lr < old_lr:
            print(f"✨ Learning rate reduced to {new_lr}. Loading best model from checkpoint.")
            generator.load_state_dict(torch.load(modelpath,weights_only=True))
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            val_loss_counter = 0
            torch.save(generator.state_dict(), modelpath)
        else:
            val_loss_counter += 1
        
        if val_loss_counter > 80:
            print("Validation loss has not improved for 20 epochs. Early stopping.")
            break

        if (epoch + 1) % 1 == 0 or epoch == 0:
            print(f"{epoch+1}/{num_epochs}] Train BCE: {avg_train_bce:.4f}, Train MSE: {avg_train_mse:.4f} | Val BCE: {avg_val_bce:.4f}, Val MSE: {avg_val_mse:.4f}")
            
    print(f"✔ Finished training for SNR={snr}, best val loss={best_val_loss:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train GAN progressively from high to low SNR.")
    parser.add_argument("--data", type=str, default="/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/", help="Path to the data directory.")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per SNR")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--n", type=int, default=2, help="scaling controll")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Training device")
    args = parser.parse_args()
    print("args.data",args.data)
    traindataset = LightCurveDataset(args.data, 'train',device=args.device)
    print('traindataset',traindataset)
    valdataset = LightCurveDataset(args.data, 'val',device=args.device)
    print(f"Total number of training samples: {len(traindataset)}")
    print(f"Total number of validation samples: {len(valdataset)}")

    traindataloader = DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valdataloader = DataLoader(valdataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    latent_dim = 120
    n = args.n

    generator = HybridConvNet(n=n)
    generator.to(args.device)
    #modelpath=f'models/mo4{n}.pth
    modelpath=f'/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/models/mo4{n}.pth'
    
    
    
    train_gan(generator, traindataloader, valdataloader, 500, num_epochs=args.epochs, device=args.device,modelpath=modelpath,n=n)
    #train_gan(generator, traindataloader, valdataloader, 50, num_epochs=args.epochs, device=args.device,modelpath=modelpath,n=n)
