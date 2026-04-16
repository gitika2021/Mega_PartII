
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

def create_noise_bins_Kepler(lc_err_arr, n=10,
                             figure_path="/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/figures/kepler_error_dist_binned.png"):
    """
    Smallest uniform bin width such that every bin has at least n samples.

    Parameters
    ----------
    lc_err_arr : 2D array-like (N,120)
        2D array containing error values
        
    n : int
        Minimum samples per bin

    Returns
    -------
    bins : ndarray
        Uniform bin edges
    width : float
        Bin width
    """

    median_err = np.sqrt(np.median(lc_err_arr**2,axis=1))
    bins = quantile_bins(median_err, n=n)
    x16 = np.percentile(median_err, 16)
    x84 = np.percentile(median_err, 84)

    fig, ax = plt.subplots(figsize=(7, 4))
    plot_binned_histogram(median_err, bins, ax=ax)
    #highlight_bin(ax, bins, bin_index, x0=x0)
    
    ax.axvline(x16,ls='--',color='yellow')
    ax.axvline(x84,ls='--',color='yellow')
    ax.tick_params(labelsize=12, length=6, width=1.5)
    plt.savefig(
    figure_path,
    dpi=500,
    bbox_inches='tight',
    pad_inches=0.2
)
    
    plt.show()

    return bins

def get_err_array(sigma_val,median_err,error_arr,bins, show=False, seed=40):
    bin_indices = [find_nearest_bin_center(val, bins) for val in sigma_val]
    #print('bin_indices',bin_indices)
    
    cnt = 0
    errors = np.zeros((len(sigma_val),error_arr.shape[1]))
    noise_multigauss = np.zeros((len(sigma_val),error_arr.shape[1]))
    noise_singlgauss = np.zeros((len(sigma_val),error_arr.shape[1]))

    rng = np.random.default_rng(42)
    for bin_index in bin_indices:
        #print('bins[bin_index]',bins[bin_index])
        err_samples = sample_k_Y_from_bin(
            median_err,error_arr,
            bins,
            bin_index,
            k=1,
            rng=rng)
        errors[cnt,:] = err_samples[0]

        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(errors.shape[1]) 
        
        noise_multigauss[cnt,:] = noise * errors[cnt,:] #generate_rowwise_gaussian_noise(errors[cnt,:], seed= 40)
        noise_singlgauss[cnt,:] = noise * np.median(errors[cnt,:])

        if show:
            plt.plot(err_samples[0],label=f'bin_index={bin_index}, med_err= {np.median(err_samples[0])}'+'\n'+f'val={sigma_val[cnt]}')
            #plt.axhline(np.median(err_samples[0]), ls='--',color='k')
            plt.legend()
            plt.show()

            plt.plot(noise_multigauss[cnt,:],label=f'multivariate gauss')
            plt.plot(noise_singlgauss[cnt,:],label=f'univariate gauss')
            #plt.axhline(np.median(err_samples[0]), ls='--',color='k')
            plt.legend()
            plt.show()
  
        cnt += 1
    return errors,noise_multigauss,noise_singlgauss

def add_noise_to_lcs(lc_file ="/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/LC10/1LC.npy",
                    kepler_lcs_error_file="/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Kepler/kepler_folded_lcs_snr50_all_binned_err.npy"):
   
    kepler_lcs_error = np.load(kepler_lcs_error_file)
    median_error = np.sqrt(np.median(kepler_lcs_error**2,axis=1))
    bins = create_noise_bins_Kepler(kepler_lcs_error, n=30)
    
    train_lcs = np.load(lc_file)
    depths = 1- np.min(train_lcs,axis=1)
    
    rng = np.random.default_rng(seed=60) 
    snr = np.random.uniform(100, 500, size= len(train_lcs))
    sigma_vals = depths/snr
    
    errors,noise_multigauss,noise_singlgauss = get_err_array(sigma_vals, median_error, kepler_lcs_error,bins,show = False);
    #print('train_lcs.shape,noise_multigauss.shape',train_lcs.shape,noise_multigauss.shape)
    lcs_noisy_multi = train_lcs + noise_multigauss

    N = train_lcs.shape[0]
 
    original_path = Path(lc_file)
    stem = original_path.stem  
    #print('stem',stem)
    new_folder = original_path.parent   
    print('new_folder',new_folder)

    time_gen = np.linspace(-1,1,train_lcs.shape[1])

    # add extra ones on both sides on lc before saving
    time_gen_ext = np.linspace(-1,1,train_lcs.shape[1]+40)
    
    for i in range(N):
        new_filename = f"{stem}{i}.npz"
        new_folder_path = os.path.join(new_folder, "Org_LC")
        os.makedirs(new_folder_path, exist_ok=True)
        
        #new_path = new_folder_path / new_filename
        new_path = new_folder /"Org_LC" / new_filename
        #print('new_path',new_path)
        # orginal lcs 
        np.savez_compressed(
            new_path,
            time=time_gen,
            flux=train_lcs[i],
            flux_err=np.zeros((len(train_lcs[i])))
        )
    
        # new_filename = f"{stem}{i}_binned.npz"
        # new_path = new_folder /"TEST_Folder" /"Binned_LC" / new_filename
        # #save_path = f"/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/LC10/TEST_Folder/{}"
        # np.savez_compressed(
        #     new_path,
        #     time=time_gen,
        #     flux=lcs_noisy_multi[i],
        #     flux_err=noise_multigauss[i]
        # )

        # extended lc files to be used for transit selection
        new_filename = f"{stem}{i}_binned.npz"
        new_folder_path = os.path.join(new_folder, "Binned_LC")
        os.makedirs(new_folder_path, exist_ok=True)
        #new_path = new_folder_path / new_filename
        
        new_path = new_folder /"Binned_LC" / new_filename
        train_lcs_ext = np.concatenate((np.ones(20)+noise_multigauss[i,0:20],lcs_noisy_multi[i],np.ones(20)+noise_multigauss[i,-20:]))
        flux_err_ext = np.concatenate((errors[i,0:20],errors[i],errors[i,-40:-20]))
        # print('train_lcs_ext.shape',train_lcs_ext.shape)
        # print('flux_err_ext.shape',flux_err_ext.shape)
        # print('time_gen_ext.shape',time_gen_ext.shape)
        np.savez_compressed(
            new_path,
            time=time_gen_ext,
            flux=train_lcs_ext,
            flux_err=flux_err_ext
        )
    return 


def main_runner(lcs_path, save_prefix,inrat, num_simulations=2, total_masks=0, ldcr_grid_path='',N=1, org_lc_path=""):
    total_lc = total_masks * num_simulations
    ldcr_grid = np.load(ldcr_grid_path)
    #print('org_lc_path',org_lc_path)
    print(f"Generating parameter sets for {total_masks} masks x {num_simulations} sims...")
    param_sets = []
    for _ in range(total_masks):
        mask_params = []
        #ratio = 10 #np.random.uniform(9, 10)
        for _ in range(num_simulations):
            ids = np.random.randint(0, ldcr_grid.shape[0], size=1)
            #ids = 734
            a = ldcr_grid[ids,0][0]
            b = ldcr_grid[ids,1][0]
            ratio = ldcr_grid[ids,3][0]
            #print('ids, a,b,ratio',ids,a,b,ratio)
            
            # a = np.random.uniform(0.2, 0.7)
            # b = np.random.uniform(0.05, 0.55)
            # ratio = inrat
            # a = 0.6676121
            # b = 0.03934647
            # ratio = 2.
            mask_params.append([a, b, ratio])
        param_sets.append(mask_params)

    task_indices = list(range(total_masks))
    #print('task_indices',task_indices)
    all_lcs_padded = np.zeros((total_lc, 200), dtype=np.float32)
    all_ld_params = np.zeros((total_lc, 3), dtype=np.float32)

    start = time.time()
    time_list = []
    flux_list = []
    flux_err_list = []
    with ProcessPoolExecutor(max_workers=32, initializer=init_worker, initargs=(maps_path, param_sets)) as executor:
        
        print(f"🚀 Starting batched simulation...")
        results_generator = tqdm(executor.map(process_mask_batch, task_indices, repeat(N), repeat(org_lc_path),chunksize=1), total=len(task_indices))
        
        for batch_result in results_generator:
            for final_idx, flux, ld_params, time_raw, flux_raw,flux_raw_err in batch_result:
                all_lcs_padded[final_idx] = flux
                all_ld_params[final_idx] = ld_params
                time_list.append(time_raw)
                flux_list.append(flux_raw)
                flux_err_list.append(flux_raw_err)


    end = time.time()
    print(f"⏱️ Time taken: {end - start:.2f} seconds")
    
    # save_dir = "/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/useless_data/"
    # save_batch_npz(save_dir, time_list, flux_list, flux_err_list)

    end = time.time()
    print(f"⏱️ Time taken: {end - start:.2f} seconds")

    print("Interpolating final results...")
    processed_lightcurves = process_all_lightcurves(all_lcs_padded, target_length=120)
    
    np.save(f"{save_prefix}LC.npy", processed_lightcurves)
    np.save(f"{save_prefix}_meta.npy", all_ld_params)
    print("✅ Saved LCs and Meta.")


if __name__ == "__main__":
    start = time.time()
    kepler_lcs_error_file1="/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Kepler/kepler_folded_lcs_snr50_all_binned_err.npy"
    lc_file1 ="/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/LC10/1LC.npy"
    add_noise_to_lcs(lc_file=lc_file1, kepler_lcs_error_file=kepler_lcs_error_file1)
    end = time.time()
    print(f"time taken is: {(end-start)/60} minutes")



