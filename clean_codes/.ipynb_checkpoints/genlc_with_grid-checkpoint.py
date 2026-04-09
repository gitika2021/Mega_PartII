import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os
import sys
import random
from itertools import repeat
# --- EightBitTransit Import ---
from EightBitTransit.TransitingImage import TransitingImage as TransitingImage

# --- Optimization Flags ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# --- Helper Functions ---
def save_batch_npz(save_dir, times_list, flux_list, flux_err_list):
    #save_path = os.path.join(self.save_dir, f"batch_{batch_idx:05d}.npz")
    save_path = os.path.join(save_dir, f"batch_{0}.npz")
    np.savez_compressed(
        save_path,
        time=np.array(times_list, dtype=object),
        flux=np.array(flux_list, dtype=object),
        flux_err=np.array(flux_err_list, dtype=object)
    )
    
def extract_and_interpolate(lc, target_length=100):
    non_zero = np.nonzero(lc)[0]
    if len(non_zero) == 0:
        return np.zeros(target_length)
    start, end = non_zero[0], non_zero[-1] + 1
    transit = lc[start:end]
    x_orig = np.linspace(0, 1, len(transit))
    x_new = np.linspace(0, 1, target_length)
    interpolator = interp1d(x_orig, transit, kind='linear')
    return interpolator(x_new)

def process_all_lightcurves(lc_array, target_length=100):
    N = lc_array.shape[0]
    processed = np.ones((N, target_length), dtype=np.float32)
    for i in range(N):
        processed[i] = extract_and_interpolate(lc_array[i], target_length)
    return processed

def simulate_one_lc(opacity_map, v=0.4, t_ref=0.0, LDlaw='quadratic', LDCs=[0.3, 0.2],
                    star2mega_radius_ratio=10, n_times=1000): 
    # NOTE: 'show' parameter removed from definition
    
    if np.max(opacity_map) > 0:
        opacity_map = opacity_map / np.max(opacity_map)
    
    radius_mega = opacity_map.shape[0] / 2
    radius_star = radius_mega * star2mega_radius_ratio
    pad = int(radius_star - radius_mega)
    if pad < 0: pad = 0
        
    padded_map = np.pad(opacity_map, pad_width=((pad, pad), (6, 6)), mode='constant', constant_values=0.0)
    times = np.linspace(-35, 35, n_times)

    model = TransitingImage(
        opacitymat=padded_map,
        v=v, t_ref=t_ref, t_arr=times,
        LDlaw=LDlaw, LDCs=LDCs
    )
    flux, overlap_times = model.gen_LC(model.t_arr)
    return overlap_times, flux

# --- Global storage ---
global_maps = None
global_param_sets = None

def init_worker(maps_path, param_sets):
    global global_maps, global_param_sets
    global_maps = np.load(maps_path, mmap_mode='r') 
    global_param_sets = param_sets

def process_mask_batch(mask_idx, N, path):
    """
    Process ALL simulations for a single mask index in one go.
    """
    # FIX 1: COPY=TRUE is required to make the array writable for Numba
    current_map = np.array(global_maps[mask_idx], copy=True)
    #print('mask_idx',mask_idx)
    #print('path',path)
    params_for_mask = global_param_sets[mask_idx]
    #print('params_for_mask',params_for_mask)
    batch_results = []
    # times = []
    # fluxes = []
    # Iterate through all 'n' simulations for this specific mask
    for sim_idx, params in enumerate(params_for_mask):
        a, b, ratio = params
        LDCs = [a, b]
        final_idx = (mask_idx * len(params_for_mask)) + sim_idx

        try:
            # FIX 2: Removed 'show=False' from arguments
            time, flux = simulate_one_lc(
                opacity_map=current_map,
                v=0.4, t_ref=0.0, LDlaw='quadratic', LDCs=LDCs,
                star2mega_radius_ratio=ratio,
                n_times=2000
            )

            padded_flux = np.zeros(200, dtype=np.float32)
            L = min(len(flux), 200)
            start_pad = (200 - L) // 2
            padded_flux[start_pad:start_pad + L] = flux[:L]
            flux_error = np.zeros(len(flux), dtype=np.float32)
        
            np.savez_compressed(
                f"{path}{N}{mask_idx}.npz",
                time=time,
                flux=flux,
                flux_err=flux_error
            )
            
            batch_results.append((final_idx, padded_flux, [a, b, ratio],time,flux,flux_error))
            # times.append(time)
            # fluxes.append(flux)
        except Exception as e:
            # Print enabled so you can see real errors in log.txt
            print(f"Error in mask {mask_idx}, sim {sim_idx}: {e}")
            batch_results.append((final_idx, np.zeros(200, dtype=np.float32), [a, b, ratio],np.zeros(200, dtype=np.float32),np.zeros(200, dtype=np.float32)))
            #batch_results_raw.append((final_idx,  np.zeros(200, dtype=np.float32),  np.zeros(200, dtype=np.float32), [a, b, ratio]))
            # times.append(np.zeros(200, dtype=np.float32))
            # fluxes.append(np.zeros(200, dtype=np.float32))

    # save_dir = "/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/useless_data/"
    # save_batch_npz(save_dir, times, fluxes)
    return batch_results

# --- Main Runner ---

def run_simulation_for_masks(maps_path, save_prefix,inrat, num_simulations=2, total_masks=0, ldcr_grid_path='',N=1, org_lc_path=""):
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
    try:
        if len(sys.argv) > 1:
            N = str(sys.argv[1])
            # n = int(sys.argv[2])
            # rsrp=int(sys.argv[3])
            # maps_path = str(sys.argv[2])
            # ldc_ratio_path = str(sys.argv[3])
            # out_dir = str(sys.argv[4])
            
            n = 1
            rsrp = 10
            maps_path = f"/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/OM10/{N}.npy"
            ldc_ratio_path = f"/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/LDC_RPRS/ldc_ratio_grid_set.npy"
            
            temp_map = np.load(maps_path, mmap_mode='r')
            total_masks = temp_map.shape[0]
            print(f"Detected {total_masks} masks in {maps_path}")

            org_lc_path0 = f"/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/LC_ORG/"
            run_simulation_for_masks(maps_path, f"/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/LC10/{N}",inrat=rsrp, num_simulations=n, total_masks=total_masks, ldcr_grid_path = ldc_ratio_path,N=N,org_lc_path=org_lc_path0)
        else:
            print('give arg: python3 genlc.py [N] [n]')
    except Exception as e:
        print(f"Main Execution Error: {e}")
