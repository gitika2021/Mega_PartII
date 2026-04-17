


import sys
# sys.path.append("../pyscripts")
# sys.path.append("../Python-Scripts")
# sys.path.append("../../../Reanalysis_Git")
# sys.path.append("../../../Reanalysis_Git")

import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from lightkurve import LightCurve, LightCurveCollection, read, search_lightcurve
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import time
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import concurrent.futures
from tqdm import tqdm
import re

class TransitRegionSelector():

    def __init__(self,ltcrv_files_folder="/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/kepler_data/phase_folded_lcs/Binned_LC/"):
        self.filesfolder = Path(ltcrv_files_folder)
        

    def find_main_dip_with_expansion(self,x, y,
                                     smooth_window=9,
                                     polyorder=2,
                                     sigma=2,
                                     min_width=0.0,
                                     expand_fraction=0.1, 
                                    transit_flux_threshold = 0.99990):
        """
        Detect strongest contiguous dip and expand region
        by a fraction of its width on both sides.
    
        expand_fraction = 0.2  → expand 20% beyond ingress/egress
        """
    
        x = np.asarray(x)
        y = np.asarray(y)
    
        # --- Safe smoothing window ---
        if smooth_window >= len(y):
            smooth_window = len(y) - 1 if len(y) % 2 == 0 else len(y)
        if smooth_window % 2 == 0:
            smooth_window += 1
    
        y_smooth = savgol_filter(y, smooth_window, polyorder)
    
        # plt.plot(y)
        # plt.plot(y_smooth)
        # plt.show()
        
        # --- Robust baseline ---
        #median = np.median(y_smooth)
        median = np.median(y_smooth[y_smooth>=transit_flux_threshold])
        #mad = np.median(np.abs(y_smooth - median))
        mad = np.median(np.abs(y_smooth - median)[y_smooth>=transit_flux_threshold])
        robust_std = 1 * mad#1.4826 * mad
        #robust_std = np.std(y_smooth[y_smooth>=transit_flux_threshold])
        threshold = median - sigma * robust_std
        #print('threshold',threshold)
        # --- Contiguous mask ---
        mask = y_smooth < threshold
        diff = np.diff(mask.astype(int))
    
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0]
    
        if mask[0]:
            starts = np.insert(starts, 0, 0)
        if mask[-1]:
            ends = np.append(ends, len(mask) - 1)
    
        if len(starts) == 0:
            return None
    
        # --- Choose deepest dip ---
        best = None
        best_depth = -np.inf
    
        for s, e in zip(starts, ends):
            width = x[e] - x[s]
            if width < min_width:
                continue
    
            segment = y_smooth[s:e+1]
            depth = median - np.min(segment)
    
            if depth > best_depth:
                best_depth = depth
                best = (s, e)
    
        if best is None:
            return None
    
        s, e = best
        width = x[e] - x[s]
    
        # --- Expand region in x-units ---
        expansion = width * expand_fraction
    
        new_start_x = x[s] - expansion
        new_end_x = x[e] + expansion
    
        # Find nearest indices
        s_exp = np.searchsorted(x, new_start_x, side="left")
        e_exp = np.searchsorted(x, new_end_x, side="right") - 1
    
        # Clamp to bounds
        s_exp = max(0, s_exp)
        e_exp = min(len(x) - 1, e_exp)
    
        segment = y_smooth[s:e+1]
        min_idx = s + np.argmin(segment)
    
        return {
            # original detected region
            "start_x": x[s],
            "end_x": x[e],
            "width": width,
            "depth": median - np.min(segment),
            "min_x": x[min_idx],
    
            # expanded region
            "expanded_start_x": x[s_exp],
            "expanded_end_x": x[e_exp],
            "expanded_width": x[e_exp] - x[s_exp],
    
            # full expanded curve
            "x_curve": x[s_exp:e_exp+1],
            "y_curve": y[s_exp:e_exp+1],
            "y_smooth_curve": y_smooth[s_exp:e_exp+1],
    
            # indices
            "start_index": s_exp,
            "end_index": e_exp
        }


    def find_transit_region_and_save_serial(self, target_ltcrv_length = 120):
        ltcrv_npz_files = list(self.filesfolder.glob("*_binned.npz"))
        for file_path in ltcrv_npz_files:
            #print(f"Loading {file_path.name}")
            data = np.load(file_path)
            self.lc_fold_load = LightCurve(time=data['time'], flux=data['flux'], flux_err=data['flux_err'])
        
            x = lc_fold_load.time.value
            y = lc_fold_load.flux.value
            yerr = lc_fold_load.flux.value   
            #print('type',type(lc_fold_load.time.value))

            self.result = self.find_main_dip_with_expansion(x, y,
                                     smooth_window=9,
                                     polyorder=2,
                                     sigma=2,
                                     min_width=0.0,
                                     expand_fraction=1.2)

            interp_func = interp1d(self.result["x_curve"], self.result["y_curve"], kind='linear', fill_value='extrapolate')
            self.xnew = np.linspace(self.result["x_curve"].min(),self.result["x_curve"].max(),target_ltcrv_length)
            self.ynew = interp_func(self.xnew)

            interp_func_err = interp1d(result["x_curve"], yerr[result["start_index"]:result["end_index"]+1], kind='linear', fill_value='extrapolate')
            ynewerr = interp_func_err(xnew)
            
            original_path = Path(file_path)
            # Extract filename without extension
            stem = original_path.stem  
            #print('stem',stem)
            new_filename = f"{stem}_transit.npz"
            new_folder = original_path.parent   
            new_path = new_folder / new_filename
            #print('new_path',new_path)
            np.savez_compressed(new_path,
                                    time=result["x_curve"],
                                    flux=result["y_curve"],
                                    flux_err=yerr[result["start_index"]:result["end_index"]+1])
        
            # original_path = Path(file_path)
            # # Extract filename without extension
            # stem = original_path.stem  
            # print('stem',stem)
            new_filename = f"{stem}_transit_interp.npz"
            new_folder = original_path.parent   
            new_path = new_folder / new_filename
           # print('new_path',new_path)
            np.savez_compressed(new_path,
                                    time=xnew,
                                    flux=ynew,
                                    flux_err=ynewerr)

    def process_one_target(self, file_path, target_ltcrv_length=120):
        #print(f"Loaded {file_path}"+"\n")
        data = np.load(file_path)
        lc_fold_load = LightCurve(time=data['time'], flux=data['flux'], flux_err=data['flux_err'])
    
        x = lc_fold_load.time.value
        y = lc_fold_load.flux.value
        yerr = lc_fold_load.flux_err.value
    
        self.result = self.find_main_dip_with_expansion(
            x, y,
            smooth_window=27,
            polyorder=2,
            sigma=2,
            min_width=0.0,
            expand_fraction=0.05
        )
    
        interp_func = interp1d(
            self.result["x_curve"],
            self.result["y_curve"],
            kind='linear',
            fill_value='extrapolate'
        )
    
        self.xnew = np.linspace(
            self.result["x_curve"].min(),
            self.result["x_curve"].max(),
            target_ltcrv_length
        )
    
        self.ynew = interp_func(self.xnew)
    
        interp_func_err = interp1d(
            self.result["x_curve"],
            yerr[self.result["start_index"]:self.result["end_index"]+1],
            kind='linear',
            fill_value='extrapolate'
        )
    
        ynewerr = interp_func_err(self.xnew)
    
        original_path = Path(file_path)
        stem = original_path.stem
        new_folder = original_path.parent
    
        # Save cropped transit
        new_filename = f"{stem}_transit.npz"
        new_path = new_folder / new_filename
    
        np.savez_compressed(
            new_path,
            time=self.result["x_curve"],
            flux=self.result["y_curve"],
            flux_err=yerr[self.result["start_index"]:self.result["end_index"]+1]
        )
    
        # Save interpolated transit
        new_filename = f"{stem}_transit_interp.npz"
        new_path = new_folder / new_filename
    
        np.savez_compressed(
            new_path,
            time=self.xnew,
            flux=self.ynew,
            flux_err=ynewerr
        )
    
    def find_transit_region_and_save_parallel(self):
    
        ltcrv_npz_files = list(self.filesfolder.glob("*_binned.npz"))
        print(f'Number of files found is{len(ltcrv_npz_files)}')
        rows = [filepath for filepath in ltcrv_npz_files]
    
        self.max_workers = 24
        print(f"Using {self.max_workers} CPU cores")
    
        tic = time.time()
    
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
    
            futures = [
                executor.submit(self.process_one_target, row)
                for row in rows
            ]
    
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print("Worker error:", e)
    
        toc = time.time()
    
        print("Total time (minutes):", (toc - tic) / 60.0)

    def load_and_plot_matched_ltcrvs(self,
        folder1,
        folder2,
        folder3,
        folder4,
        pattern="kplr*.npz",
        x_key="x",
        y_key="y",
        show_plot=True,
        save_dir=None,
        N_plots = None
                                    
    ):
        """
        Load matching NPZ files from two folders and plot them together.
    
        Parameters
        ----------
        folder1 : str
            Path to first folder.
        folder2 : str
            Path to second folder.
        pattern : str
            Glob pattern for matching files.
        x_key : str
            Key name for x data inside npz file.
        y_key : str
            Key name for y data inside npz file.
        show_plot : bool
            Whether to display plots.
        save_dir : str or None
            If provided, saves plots to this directory.
        """
    
        def extract_key(filepath,split_str="_binned.npz"):
            filename = os.path.basename(filepath)
            #filename.split("_binned.npz")[0]
            return filename.split(split_str)[0]
    
        files1 = glob.glob(os.path.join(folder1, pattern))
        files2 = glob.glob(os.path.join(folder2, pattern))
        files3 = glob.glob(os.path.join(folder3, pattern))
        files4 = glob.glob(os.path.join(folder3, pattern))
        grouped = defaultdict(dict)
        print('files1',files1,len(files1))
        print('files2',files2,len(files2))
        print('files3',files3,len(files3))
        print('files4',files4,len(files4))
        for f in files1:
            grouped[extract_key(f,split_str=".npz")]["folder1"] = f #extract_key(f,split_str=".npz")
    
        for f in files2:
            grouped[extract_key(f,split_str="_binned.npz")]["folder2"] = f #extract_key(f,split_str="_binned.npz")
            
        for f in files3:
            grouped[extract_key(f,split_str="_binned_transit.npz")]["folder3"] = f #extract_key(f,split_str="_binned.npz")
    
        for f in files4:
            grouped[extract_key(f,split_str="_binned_transit_interp.npz")]["folder4"] = f #extract_key(f,split_str="_binned.npz")        
        print('grouped',list(grouped.items())[0:2])
        for i, (key, filepair ) in enumerate(grouped.items()):
            # if i >= N_plots and N_plots is not None:
            #     break
            #print(key, value)
        
        #for key, filepair in grouped.items():
            #print('filepair',filepair)
            #print(H)
            if "folder1" in filepair and "folder2" in filepair and "folder3" in filepair and "folder4" in filepair:
    
                data1 = np.load(filepair["folder1"])
                data2 = np.load(filepair["folder2"])
    
                data3 = np.load(filepair["folder3"])
                data4 = np.load(filepair["folder4"])
                
                data1_load = LightCurve(time=data1['time'], flux=data1['flux'], flux_err=data1['flux_err'])
                data2_load = LightCurve(time=data2['time'], flux=data2['flux'], flux_err=data2['flux_err'])
    
                data3_load = LightCurve(time=data3['time'], flux=data3['flux'], flux_err=data3['flux_err'])
                data4_load = LightCurve(time=data4['time'], flux=data4['flux'], flux_err=data4['flux_err'])
                
                x1, y1 = data1_load.time.value, data1_load.flux.value#data1[x_key], data1[y_key]
                x2, y2 = data2_load.time.value, data2_load.flux.value #data2[x_key], data2[y_key]
    
                x3, y3 = data3_load.time.value, data3_load.flux.value#data1[x_key], data1[y_key]
                x4, y4 = data4_load.time.value, data4_load.flux.value #data2[x_key], data2[y_key]
                
                # plt.figure()
                # plt.plot(x1, y1, label="Folder 1")
                # plt.plot(x2, y2, label="Folder 2")
                # plt.title(key)
                # plt.legend()
                fontsize=6
                fig, ax = plt.subplots(3, 1, figsize=(8,6))
                ax[0].scatter(x1, y1, color='k',s=1, label='Folded LC')
                ax[0].scatter(x2, y2, color='r',s=1, label='Binned LC')
                ax[0].set_title(f'{key}')
                ax[0].legend(fontsize=fontsize,loc="lower left")
        
                ax[1].scatter(x2, y2, color='r',s=1, label='Binned LC')
                ax[1].scatter(x3,y3,color='blue',s=1, label='Transit Region')
                ax[1].legend(fontsize=fontsize,loc="lower left")
    
                  
                ax[2].scatter(x3,y3,color='blue',s=1, label='Transit Region')
                ax[2].scatter(x4,y4,color='orange',s=5, label='Transit Region Interp')
                ax[2].legend(fontsize=fontsize,loc="lower left")
        
                #ax[0].set_title(f"Phase Folded LC: {kepname}", fontsize=8)
                ax[2].set_xlabel('Phase')
                plt.tight_layout()
            
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"{key}.png")
                    plt.savefig(save_path)
    
                if show_plot:
                    plt.show()
                else:
                    plt.close()
    
            else:
                print(f"Skipping unmatched key: {key}")
            

def extract_index(filepath):
    # Extract number after 'LC'
    match = re.search(r'LC(\d+)', filepath.name)
    if match:
        return int(match.group(1))
    return -1  # fallback if something unexpected

def combine_flux(folder_path, output_file="combined_flux.npy", savefolder_path=""):
    folder = Path(folder_path)
    savefolder = Path(savefolder_path)
    # Get all matching files
    files = list(folder.glob("*_binned_transit_interp.npz"))

    if not files:
        print("No files found!")
        return

    # Sort files based on LC index
    files_sorted = sorted(files, key=extract_index)

    print(f"Found {len(files_sorted)} files")

    all_flux = []

    for f in files_sorted:
        try:
            data = np.load(f)

            # 🔴 IMPORTANT: adjust key name if needed
            # check keys using: print(data.files)
            flux = data["flux"]  

            if flux is None:
                print(f"Skipping {f} (flux is None)")
                continue

            all_flux.append(flux)

        except Exception as e:
            print(f"Error loading {f}: {e}")

    # Convert to array
    combined = np.array(all_flux)

    print("Final shape:", combined.shape)

    # Save to .npy
    np.save(savefolder / output_file, combined)

    print(f"Saved to {output_file}")
    
if __name__ == "__main__":
    # folder="/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/kepler_data/phase_folded_lcs/Binned_LC/"
    # obj = TransitRegionSelector(ltcrv_files_folder=folder)
    # obj.find_transit_region_and_save_parallel()
    # #trans = TransitRegionSelector()
    # folder1 = "/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/kepler_data/phase_folded_lcs/"
    # folder2 = "/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/kepler_data/phase_folded_lcs/Binned_LC/"
    # folder3 = "/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/kepler_data/phase_folded_lcs/Binned_LC/"
    # folder4 = "/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/kepler_data/phase_folded_lcs/Binned_LC/"
    
    # # #trans.find_transit_region_and_save(target_ltcrv_length = 120)
    # # trans.find_transit_region_and_save_parallel()
    # obj.load_and_plot_matched_ltcrvs(
    #     folder1=folder1,
    #     folder2=folder2,
    #     folder3=folder3,
    #     folder4=folder4,
    #     x_key="time",
    #     y_key="flux",
    #     save_dir="/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/kepler_data/lightcurve_plots/",
    #     show_plot=True,
    #     N_plots = 10
    # )

    start = time.time()
    folder="/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/LC10/Binned_LC/"
    obj = TransitRegionSelector(ltcrv_files_folder=folder)
    obj.find_transit_region_and_save_parallel()
    end = time.time()
    print(f"time taken is: {(end-start)/60} minutes")
    #trans = TransitRegionSelector()
    # folder1 = "/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/useless_data/org/"
    # folder2 = "/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/useless_data/binned/"
    # folder3 = "/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/useless_data/transit/"
    # folder4 = "/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/useless_data/interp/"
    
    # # #trans.find_transit_region_and_save(target_ltcrv_length = 120)
    # # trans.find_transit_region_and_save_parallel()
    # obj.load_and_plot_matched_ltcrvs(
    #     folder1=folder1,
    #     folder2=folder2,
    #     folder3=folder3,
    #     folder4=folder4,
    #     x_key="time",
    #     y_key="flux",
    #     save_dir="/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/kepler_data/lightcurve_plots/",
    #     show_plot=True,
    #     N_plots = 50,
    #     pattern="lc*_noisy*.npz"
    # )

    # save the transit region lcs to single file
    combine_flux(folder, output_file="1LC_hscaled.npy",
             savefolder_path="/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/LC10/")

    #---------------------------------------------------------------------------------------------------------------------------------
    start = time.time()
    folder="/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/LC20/Binned_LC/"
    obj = TransitRegionSelector(ltcrv_files_folder=folder)
    obj.find_transit_region_and_save_parallel()
    end = time.time()
    print(f"time taken is: {(end-start)/60} minutes")

    combine_flux(folder, output_file="1LC_hscaled.npy",
             savefolder_path="/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/LC20/")
    
