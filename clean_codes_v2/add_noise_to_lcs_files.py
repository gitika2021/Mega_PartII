import numpy as np
import os
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from tqdm import tqdm


import sys
# sys.path.append("/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/pyscripts")
# sys.path.append("/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git")
# sys.path.append("/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Python-Scripts")

import numpy as np
import matplotlib.pyplot as plt
from kepler_noise_sampler import *
import time
from lightkurve import LightCurve, LightCurveCollection, read, search_lightcurve
from pathlib import Path
import os 

# -------- GLOBALS (set inside workers) --------
kepler_lcs_error = None
median_error = None
bins = None

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

    # fig, ax = plt.subplots(figsize=(7, 4))
    # plot_binned_histogram(median_err, bins, ax=ax)
    # #highlight_bin(ax, bins, bin_index, x0=x0)
    
    # ax.axvline(x16,ls='--',color='yellow')
    # ax.axvline(x84,ls='--',color='yellow')
    # ax.tick_params(labelsize=12, length=6, width=1.5)
    # plt.savefig(
    # figure_path,
    # dpi=500,
    # bbox_inches='tight',
    # pad_inches=0.2)
    
    # plt.show()

    return bins

def get_err_array(sigma_val,median_err,error_arr,bins, show=False, seed=None):
    bin_indices = [find_nearest_bin_center(val, bins) for val in sigma_val]
    #print('bin_indices',bin_indices)
    
    cnt = 0
    errors = np.zeros((len(sigma_val),error_arr.shape[1]))
    noise_multigauss = np.zeros((len(sigma_val),error_arr.shape[1]))
    noise_singlgauss = np.zeros((len(sigma_val),error_arr.shape[1]))
    noise_gaussian = np.zeros((len(sigma_val),error_arr.shape[1]))
    
    rng = np.random.default_rng(seed)
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
        noise_gaussian[cnt,:] = noise * sigma_val[cnt]

        # if show:
        #     plt.plot(err_samples[0],label=f'bin_index={bin_index}, med_err= {np.median(err_samples[0])}'+'\n'+f'val={sigma_val[cnt]}')
        #     #plt.axhline(np.median(err_samples[0]), ls='--',color='k')
        #     plt.legend()
        #     plt.show()

        #     plt.plot(noise_multigauss[cnt,:],label=f'multivariate gauss')
        #     plt.plot(noise_singlgauss[cnt,:],label=f'univariate gauss')
        #     #plt.axhline(np.median(err_samples[0]), ls='--',color='k')
        #     plt.legend()
        #     plt.show()
  
        cnt += 1
    return errors,noise_multigauss,noise_singlgauss
    

# -------- INITIALIZER (runs once per worker) --------
def init_worker(kepler_file,figure_path):
    global kepler_lcs_error, median_error, bins

    kepler_lcs_error = np.load(kepler_file, mmap_mode='r')
    median_error = np.sqrt(np.median(kepler_lcs_error**2, axis=1))
    bins = create_noise_bins_Kepler(kepler_lcs_error, n=30,figure_path=figure_path)


# -------- WORKER FUNCTION --------
def process_lc_file(lc_file, N_dummy, org_lc_path, noise_flag ='real', seed=None,
                   snr_min=100,snr_max=500):

    train_lcs = np.load(lc_file)

    depths = 1 - np.min(train_lcs, axis=1)

    rng = np.random.default_rng(seed)
    snr = rng.uniform(snr_min, snr_max, size=len(train_lcs))
    #print('snr[0:10]',snr[0:10])
    sigma_vals = depths / snr
    print('sigma_vals[0:10]',sigma_vals[0:10])
    
    # errors, noise_multigauss, noise_gaussian = get_err_array(
    #     sigma_vals, median_error, kepler_lcs_error, bins, show=False, seed=seed
    # )
    # print('noise_multigauss[0,0:5]',noise_multigauss[0,0:5])
    # print('noise_multigauss[1,0:5]',noise_multigauss[1,0:5])
    if noise_flag == "real":
        errors, noise_multigauss, noise_gaussian = get_err_array(sigma_vals,
                                                                 median_error,
                                                                 kepler_lcs_error, bins,
                                                                 show=False, seed=seed
                                                                )
        lcs_noisy_multi = train_lcs + noise_multigauss
    elif noise_flag == "gaussian":
        noise_gaussian = np.zeros(train_lcs.shape)
        errors = np.zeros(train_lcs.shape)
        # errors = np.array([errors[i,:] = sigma_vals[i] for in range(train_lcs.shape[0]) )
        # noise_singlgauss = np.zeros((len(sigma_val),train_lcs.shape[1]))
        
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(train_lcs.shape[1]) 
        for i in range(train_lcs.shape[0]):
            noise_gaussian[i,:] = noise * sigma_vals[i]
            errors[i,:] = sigma_vals[i]
                
        lcs_noisy_multi = train_lcs + noise_gaussian
        noise_multigauss = noise_gaussian
    print(f"Added {noise_flag} noise")    
    original_path = Path(lc_file)
    stem = original_path.stem
    new_folder = original_path.parent

    np.save(new_folder / f"{stem}_noise.npy",noise_multigauss)
    np.save(new_folder / f"{stem}_snr_added.npy",snr)
    
    time_gen = np.linspace(-1, 1, train_lcs.shape[1])
    time_gen_ext = np.linspace(-1, 1, train_lcs.shape[1] + 40)

    # org_folder = new_folder / "Org_LC"
    bin_folder = new_folder / "Binned_LC"
    #os.makedirs(org_folder, exist_ok=True)
    #os.makedirs(bin_folder, exist_ok=True)

    # if not os.path.exists(org_folder):
    #            os.mkdir(org_folder)

    if not os.path.exists(bin_folder):
               os.mkdir(bin_folder)

    for i in range(train_lcs.shape[0]):

        # np.savez_compressed(
        #     org_folder / f"{stem}{i}.npz",
        #     time=time_gen,
        #     flux=train_lcs[i],
        #     flux_err=np.zeros(len(train_lcs[i]))
        # )

        train_lcs_ext = np.concatenate((
            np.ones(20) + noise_multigauss[i, 0:20],
            lcs_noisy_multi[i],
            np.ones(20) + noise_multigauss[i, -20:]
        ))

        flux_err_ext = np.concatenate((
            errors[i, 0:20],
            errors[i],
            errors[i, -40:-20]
        ))

        np.savez_compressed(
            bin_folder / f"{stem}{i}_binned.npz",
            time=time_gen_ext,
            flux=train_lcs_ext,
            flux_err=flux_err_ext
        )

    return lc_file  # useful for logging


# -------- DRIVER --------
def run_parallel(lc_files, kepler_file, max_workers=32, noise='real',figure_path=None,
                random_seed=None, snr_min=100,snr_max=500):

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,
        initargs=(kepler_file,figure_path)
    ) as executor:

        results = list(
            tqdm(
                executor.map(
                    process_lc_file,
                    lc_files,
                    repeat(None),
                    repeat(None),
                    repeat(noise),
                    repeat(random_seed),
                    repeat(snr_min),
                    repeat(snr_max),
                    chunksize=1
                ),
                total=len(lc_files)
            )
        )

    return results

def main(lc_file,kepler_error_file,figure_path,random_seed=None, snr_min=100,snr_max=500,noise='gaussian'):
    # run_parallel([lc_file],kepler_error_file,noise='real',max_workers=1,figure_path=figure_path,
    #             random_seed=random_seed)

    run_parallel([lc_file],kepler_error_file,max_workers=1,figure_path=figure_path,
                random_seed=random_seed, snr_min=snr_min,snr_max=snr_max,noise=noise)
    return


if __name__ == "__main__":
    #lc_files = np.loadtxt("lc_files.txt", dtype=str)
    basepath = f"/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/LC20/"

    
    file1 = f"{basepath}test1/1LC.npy"
    file2 = f"{basepath}test2/10LC.npy"
    print('file1',file1)
    lc_files = np.array([file1,file2], dtype=str)
    run_parallel(
        lc_files,
        kepler_file="/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Kepler/kepler_folded_lcs_snr50_all_binned_err.npy",
        noise='real',
        max_workers=len(lc_files)
    )

