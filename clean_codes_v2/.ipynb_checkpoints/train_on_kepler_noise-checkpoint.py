import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import DataLoader
import numpy as np
from datasetv2 import *
from models import *
import cProfile
import pstats
from utils import *
import matplotlib.pyplot as plt
import os
from pathlib import Path

criterionb = symmetry_aware_bce
criterion = symmetry_aware_dice_loss
rng = np.random.default_rng(seed=42)
# figpath = f"/data/project/hpc2601012/Gitika/plots/debug_sigmoid_epoch"


def save_checkpoint(epoch, generator, optimizer_G, scheduler, best_val_loss, val_loss_counter, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': generator.state_dict(),
        'optimizer_state_dict': optimizer_G.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'val_loss_counter': val_loss_counter,
    }, checkpoint_path)


def load_checkpoint(checkpoint_path, generator, optimizer_G, scheduler, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    generator.load_state_dict(ckpt['model_state_dict'])
    optimizer_G.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch       = ckpt['epoch'] + 1
    best_val_loss     = ckpt['best_val_loss']
    val_loss_counter  = ckpt['val_loss_counter']
    print(f"✔ Resumed from checkpoint at epoch {ckpt['epoch']+1}, best_val_loss={best_val_loss:.4f}")
    return start_epoch, best_val_loss, val_loss_counter


def train_gan(generator, traindataloader, valdataloader, snr,
              num_epochs=50, device="cuda", modelpath='Linear',
              n=1, resume=False, checkpoint_freq=1,figpath=None):

    optimizer_G = optim.AdamW(generator.parameters(), lr=1e-3, weight_decay=1e-8)
    scheduler   = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, 'min', patience=15, factor=0.5)

    best_val_loss    = float("inf")
    val_loss_counter = 0
    start_epoch      = 0

    # Derived paths
    best_model_path  = modelpath                                      # best weights only
    checkpoint_path  = modelpath.replace('.pth', '_checkpoint.pth')  # full resumable checkpoint

    # ── Resume from checkpoint if requested ────────────────────────────────────
    if resume and os.path.exists(checkpoint_path):
        start_epoch, best_val_loss, val_loss_counter = load_checkpoint(
            checkpoint_path, generator, optimizer_G, scheduler, device
        )
    elif resume:
        print(f"⚠ No checkpoint found at {checkpoint_path}. Starting from scratch.")

    train_loss_bce = []
    train_loss_mse = []

    val_loss_bce = []
    val_loss_mse = []

    epochs_arr = []
    # ── Training loop ───────────────────────────────────────────────────────────
    for epoch in range(start_epoch, num_epochs):
        generator.train()
        gbce_total = 0
        gmse_total = 0

        for i, (lc_batch, real_depths, real_imgs) in enumerate(traindataloader):
            lc_input = lc_batch.squeeze(1)
            optimizer_G.zero_grad()
            gen_imgs = generator(lc_input.view(lc_input.shape[0], 1, 120))

            g_loss_bce = symmetry_aware_bce(real_imgs.squeeze(), gen_imgs.squeeze())
            g_loss_mse = symmetry_aware_mse(real_imgs.squeeze(), gen_imgs.squeeze())

            genlos = g_loss_bce  # kept consistent with original logic

            gmse_total += g_loss_mse.detach()
            gbce_total += g_loss_bce.detach()

            genlos.backward()
            if i == 1 and epoch % 10 == 0:
                print_grad_stats(generator, i)
            optimizer_G.step()

            # ── Debug plots every 10 epochs ─────────────────────────────────────
            if i == 0 and epoch % 10 == 0:
                prob_map   = gen_imgs[0].squeeze().detach().cpu().numpy()
                true_img   = real_imgs[0].squeeze().detach().cpu().numpy()
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

                if figpath is not None:
                    plt.savefig(f"{figpath}/epoch{epoch}.png")
                else:
                    plt.show()
                plt.close()

        avg_train_bce = gbce_total / len(traindataloader)
        avg_train_mse = gmse_total / len(traindataloader)

        train_loss_bce.append(avg_train_bce)
        train_loss_mse.append(avg_train_mse)
        epochs_arr.append(epoch)
        # ── Validation ──────────────────────────────────────────────────────────
        val_bce_losses   = []
        val_mse_losses   = []
        val_total_losses = []

        generator.eval()
        with torch.no_grad():
            for lc_batch, real_depths, real_imgs in valdataloader:
                lc_input = lc_batch.squeeze(1)
                gen_imgs = generator(lc_input.view(lc_input.shape[0], 1, 120))

                val_bce = symmetry_aware_bce(real_imgs.squeeze(), gen_imgs.squeeze())
                val_mse = symmetry_aware_mse(real_imgs.squeeze(), gen_imgs.squeeze())

                val_bce_losses.append(val_bce.item())
                val_mse_losses.append(val_mse.item())
                val_total_losses.append(val_bce.item() + val_mse.item())

        avg_val_bce  = np.mean(val_bce_losses)
        avg_val_mse  = np.mean(val_mse_losses)
        avg_val_loss = np.mean(val_total_losses)

        val_loss_bce.append(avg_val_bce)
        val_loss_mse.append(avg_val_mse)
        
        old_lr = optimizer_G.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer_G.param_groups[0]['lr']

        if new_lr < old_lr:
            print(f"✨ LR reduced to {new_lr}. Loading best model weights from {best_model_path}.")
            generator.load_state_dict(torch.load(best_model_path, weights_only=True))

        # ── Save best weights ────────────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss    = avg_val_loss
            val_loss_counter = 0
            torch.save(generator.state_dict(), best_model_path)
            print(f"  ✔ New best model saved (val_loss={best_val_loss:.4f})")
        else:
            val_loss_counter += 1

        # ── Save full resumable checkpoint every N epochs ───────────────────────
        if (epoch + 1) % checkpoint_freq == 0 or (epoch + 1) == num_epochs:
            save_checkpoint(epoch, generator, optimizer_G, scheduler,
                            best_val_loss, val_loss_counter, checkpoint_path)
            print(f"  💾 Checkpoint saved at epoch {epoch+1}")

        # ── Early stopping ───────────────────────────────────────────────────────
        if val_loss_counter > 80:
            print("Validation loss has not improved for 80 epochs. Early stopping.")
            break

        if (epoch + 1) % 1 == 0 or epoch == 0:
            print(f"[{epoch+1}/{num_epochs}] "
                  f"Train BCE: {avg_train_bce:.4f}, Train MSE: {avg_train_mse:.4f} | "
                  f"Val BCE: {avg_val_bce:.4f}, Val MSE: {avg_val_mse:.4f}")


            # plot training and validation loss
        plt.figure(figsize=(5, 5))
    
        plt.subplot(1, 1, 1)
        plt.title("")
        plt.plot(epochs_arr, train_loss_mse, '.-k',label='train mse' )
        plt.plot(epochs_arr, train_loss_bce, '.--k',label='train bce' )
    
        plt.plot(epochs_arr, val_loss_mse, '.-b',label='val mse' )
        plt.plot(epochs_arr, val_loss_bce, '.--b',label='val bce' )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
       
    
        if figpath is not None:
            dygnostic_dir = Path(figpath) / "dygnostics"
            dygnostic_dir.mkdir(parents=True, exist_ok=True)
            losses = np.zeros((len(train_loss_mse),5))
            losses[:,0] = epochs_arr
            losses[:,1] = train_loss_mse
            losses[:,2] = train_loss_bce
            losses[:,3] = val_loss_mse
            losses[:,4] = val_loss_bce
            np.save(f"{str(dygnostic_dir)}/epoch_vs_loss.npy",losses)
            plt.savefig(f"{figpath}/epoch_vs_loss.png")
        else:
            plt.show()
        plt.close()
        
    print(f"✔ Finished training for SNR={snr}, best val loss={best_val_loss:.4f}")

    
    

def main(data_dir,model_dir,epochs,batch_size,n_scale,device,resume,checkpoint_freq,figpath=None):
    traindataset = LightCurveDataset(data_dir, 'train', device=device)
    valdataset   = LightCurveDataset(data_dir, 'val',   device=device)
    print(f"Total number of training samples:   {len(traindataset)}")
    print(f"Total number of validation samples: {len(valdataset)}")

    traindataloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    valdataloader   = DataLoader(valdataset,   batch_size=batch_size, shuffle=False, num_workers=0)

    generator = HybridConvNet(n=n_scale)
    generator.to(device)

    modelpath = f"{model_dir}/model_n{n_scale}.pth" # change this

    train_gan(generator, traindataloader, valdataloader, 500,
              num_epochs=epochs, device=device,
              modelpath=modelpath, n=n_scale, resume=resume,
              checkpoint_freq=checkpoint_freq,figpath=figpath)

    return
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train GAN progressively from high to low SNR.")
    parser.add_argument("--data",       type=str, default="/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/", help="Path to the data directory.")
    parser.add_argument("--epochs",     type=int, default=10,  help="Epochs per SNR")
    parser.add_argument("--batch-size", type=int, default=32,  help="Batch size for training")
    parser.add_argument("--n",          type=int, default=2,   help="Scaling control")
    parser.add_argument("--device",     type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Training device")
    parser.add_argument("--modelpath",  type=str, required=True, help="Directory to save model files")
    parser.add_argument("--resume",          action="store_true", help="Resume training from checkpoint if available")
    parser.add_argument("--checkpoint-freq", type=int, default=1, help="Save a resumable checkpoint every N epochs (default: 1)")
    args = parser.parse_args()

    print("args.data", args.data)

    traindataset = LightCurveDataset(args.data, 'train', device=args.device)
    valdataset   = LightCurveDataset(args.data, 'val',   device=args.device)
    print(f"Total number of training samples:   {len(traindataset)}")
    print(f"Total number of validation samples: {len(valdataset)}")

    traindataloader = DataLoader(traindataset, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    valdataloader   = DataLoader(valdataset,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    n         = args.n
    generator = HybridConvNet(n=n)
    generator.to(args.device)

    modelpath = f"{args.modelpath}mo4{n}.pth"

    train_gan(generator, traindataloader, valdataloader, 500,
              num_epochs=args.epochs, device=args.device,
              modelpath=modelpath, n=n, resume=args.resume,
              checkpoint_freq=args.checkpoint_freq)




# import torch
# import torch.nn as nn
# import torch.optim as optim
# import json
# from torch.utils.data import DataLoader
# import numpy as np
# from datasetv2 import *
# #from models2 import *
# from models import *
# import cProfile
# import pstats
# from utils import *
# import matplotlib.pyplot as plt
# # In main.py
# #criterion = nn.MSELoss()
# criterionb = symmetry_aware_bce

# criterion = symmetry_aware_dice_loss
# rng = np.random.default_rng(seed=42) # Optional: provide a seed for reproducibility
# figpath = f"/data/project/hpc2601012/Gitika/plots/debug_sigmoid_epoch"
# def train_gan(generator, traindataloader, valdataloader, snr, num_epochs=50, device="cuda",modelpath='Linear',n=1):
#     optimizer_G = optim.AdamW(generator.parameters(), lr=1e-3, weight_decay=1e-8)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, 'min', patience=15, factor=0.5)
#     best_val_loss = float("inf") # Simplified this
#     val_loss_counter = 0

#     for epoch in range(num_epochs):
#         generator.train()
#         gbce_total = 0
#         gmse_total = 0
#         # if epoch>100:
#         #     snru=50+450*rng.random()
#         # elif epoch>60:
#         #     snru=250+250*rng.random()
#         # elif epoch>30:
#         #     snru=350+150*rng.random()
#         # else:
#         #     snru=400+100*rng.random()
#         for i, (lc_batch, real_depths, real_imgs) in enumerate(traindataloader):
#             # noisy_lc = add_noise_to_batch(lc_batch.squeeze(1), snru)
#             # #print(noisy_lc.shape)
#             # optimizer_G.zero_grad()
#             # gen_imgs = generator(noisy_lc.view(noisy_lc.shape[0],1,120))

#             lc_input = lc_batch.squeeze(1)
#             optimizer_G.zero_grad()
#             gen_imgs = generator(lc_input.view(lc_input.shape[0], 1, 120))
            
#             g_loss_bce = symmetry_aware_bce(real_imgs.squeeze(),gen_imgs.squeeze()) 
            
#             g_loss_mse = symmetry_aware_mse(real_imgs.squeeze(),gen_imgs.squeeze())
#             if epoch>60:
#                 genlos = g_loss_bce
#             else:
#                 genlos = g_loss_bce
            
#             gmse_total += g_loss_mse.detach()
#             gbce_total += g_loss_bce.detach()
            
#             genlos.backward()
#             if i == 1 and epoch%10==0:
#                 print_grad_stats(generator, i)
#             optimizer_G.step()
#             if i == 0 and epoch % 10 == 0:
                

#                 prob_map = gen_imgs[0].squeeze().detach().cpu().numpy()
#                 true_img = real_imgs[0].squeeze().detach().cpu().numpy()

#                 hard_shape = (prob_map > 0.5).astype(float)

#                 plt.figure(figsize=(12, 4))

#                 plt.subplot(1, 3, 1)
#                 plt.title("Ground Truth")
#                 plt.imshow(true_img, vmin=0, vmax=1, cmap='inferno')
#                 plt.axis('off')

#                 plt.subplot(1, 3, 2)
#                 plt.title(f"Sigmoid Output (Confidence)\nMSE: {g_loss_mse.item():.4f}")
#                 plt.imshow(prob_map, vmin=0, vmax=1, cmap='inferno')
#                 plt.axis('off')

#                 plt.subplot(1, 3, 3)
#                 plt.title("Thresholded > 0.5\n(Final Prediction)")
#                 plt.imshow(hard_shape, vmin=0, vmax=1, cmap='inferno')
#                 plt.axis('off')

#                 #plt.savefig(f"plots/debug_sigmoid_epoch_{epoch}.png")
#                 plt.savefig(f"{figpath}_{epoch}.png")
#                 plt.close()

#         avg_train_bce = gbce_total / len(traindataloader)
#         avg_train_mse = gmse_total / len(traindataloader)
        
#         val_bce_losses = []
#         val_mse_losses = []
#         val_total_losses = []
        
#         generator.eval() 
#         with torch.no_grad():
#             for lc_batch, real_depths, real_imgs in valdataloader:
#                 # noisy_lc = add_noise_to_batch(lc_batch.squeeze(1), snru)
#                 # gen_imgs = generator(noisy_lc.view(noisy_lc.shape[0],1,120))

#                 lc_input = lc_batch.squeeze(1)
#                 gen_imgs = generator(lc_input.view(lc_input.shape[0], 1, 120))
            
#                 val_bce = symmetry_aware_bce(real_imgs.squeeze(), gen_imgs.squeeze())
#                 val_mse = symmetry_aware_mse(real_imgs.squeeze(), gen_imgs.squeeze())
                
#                 val_bce_losses.append(val_bce.item())
#                 val_mse_losses.append(val_mse.item())
#                 if epoch>60:
#                     val_total_losses.append(val_mse.item()+val_bce.item())
#                 else:
#                     val_total_losses.append(val_bce.item()+val_mse.item())
        
#         avg_val_bce = np.mean(val_bce_losses)
#         avg_val_mse = np.mean(val_mse_losses)
#         avg_val_loss = np.mean(val_total_losses) 
#         old_lr = optimizer_G.param_groups[0]['lr'] 
        
#         scheduler.step(avg_val_loss)
        
#         new_lr = optimizer_G.param_groups[0]['lr']        
#         if new_lr < old_lr:
#             print(f"✨ Learning rate reduced to {new_lr}. Loading best model from checkpoint.")
#             generator.load_state_dict(torch.load(modelpath,weights_only=True))
        
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             val_loss_counter = 0
#             torch.save(generator.state_dict(), modelpath)
#         else:
#             val_loss_counter += 1
        
#         if val_loss_counter > 80:
#             print("Validation loss has not improved for 20 epochs. Early stopping.")
#             break

#         if (epoch + 1) % 1 == 0 or epoch == 0:
#             print(f"{epoch+1}/{num_epochs}] Train BCE: {avg_train_bce:.4f}, Train MSE: {avg_train_mse:.4f} | Val BCE: {avg_val_bce:.4f}, Val MSE: {avg_val_mse:.4f}")
            
#     print(f"✔ Finished training for SNR={snr}, best val loss={best_val_loss:.4f}")

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Train GAN progressively from high to low SNR.")
#     parser.add_argument("--data", type=str, default="/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/", help="Path to the data directory.")
#     parser.add_argument("--epochs", type=int, default=10, help="Epochs per SNR")
#     parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
#     parser.add_argument("--n", type=int, default=2, help="scaling controll")
#     parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Training device")
#     parser.add_argument("--modelpath", type=str, required=True, help=f'/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/models/')
#     args = parser.parse_args()
#     print("args.data",args.data)
#     traindataset = LightCurveDataset(args.data, 'train',device=args.device)
#     print('traindataset',traindataset)
#     valdataset = LightCurveDataset(args.data, 'val',device=args.device)
#     print(f"Total number of training samples: {len(traindataset)}")
#     print(f"Total number of validation samples: {len(valdataset)}")

#     traindataloader = DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
#     valdataloader = DataLoader(valdataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

#     latent_dim = 120
#     n = args.n

#     generator = HybridConvNet(n=n)
#     generator.to(args.device)
#     #modelpath=f'models/mo4{n}.pth
#     #modelpath=f'/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/models/mo4{n}.pth'
#     modelpath = args.modelpath
#     modelpath = f"{modelpath}mo4{n}.pth"
    
#     train_gan(generator, traindataloader, valdataloader, 500, num_epochs=args.epochs, device=args.device,modelpath=modelpath,n=n)
#     #train_gan(generator, traindataloader, valdataloader, 50, num_epochs=args.epochs, device=args.device,modelpath=modelpath,n=n)
