import numpy as np
import matplotlib.pyplot as plt
from class_modules import MLPreProcessing, LogBinHistogram
from paths import *
from koi_table import KoiTableObjs as koitab
import pandas as pd
from pathlib import Path
from utils import extract_key
import ast
import shutil

def remove_nan_from_arrays(a, b, c):
    """
    Remove entries where any of the arrays contains NaN.
    Keeps indices aligned across arrays.
    """
    
    mask = ~(np.isnan(a) | np.isnan(b) | np.isnan(c))
    
    a_clean = a[mask]
    b_clean = b[mask]
    c_clean = c[mask]
    
    return a_clean, b_clean, c_clean,mask
    
def save_kepler_ldc_ratio(koi_table_folder = "/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Kepler/",
                          koi_table_filename = "koi_cumulative_2025.06.28_01.24.15.csv",
                         ldc_ratio_outfile = "/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Kepler/kepler_ldc_coeffs_conf_planets.npy",
                         snr_cut=50):
    """
    kepler koi table file: koi_cumulative_2025.06.28_01.24.15.csv should be downloaded first and saved in 'koi_table_file' directory
    ldc_ratio_outfile: (N,a,b,rp/rs) from kepler koi table after removing nans etc.
    """
    koi = koitab(files_dir = koi_table_folder,koi_file_name = koi_table_filename,verbose=False)
    koi_table = koi.koi_table
    #print(koi_table)
    koi.get_koi_confirmed()
    koi_conf_plans_tabl = koi.koi_conf_plans_tab
    
    df_new =koi_conf_plans_tabl # koi_table # 
    kepler_lcs_rprs = df_new["koi_ror"].to_numpy()
    kepler_lcs_ldca = df_new["koi_ldm_coeff1"].to_numpy()
    kepler_lcs_ldcb = df_new["koi_ldm_coeff2"].to_numpy()
    kepler_snr = df_new["koi_model_snr"].to_numpy()
    
    ldca, ldcb, rprs,mask = remove_nan_from_arrays(kepler_lcs_ldca, kepler_lcs_ldcb, kepler_lcs_rprs)
    kepler_snr = kepler_snr[mask]
    print('Number of valid values in Kepler KOI for LDC and Rp/Rs',len(ldca))
    #np.isnan(a).any(),np.isnan(a).sum()
    LDC_coeffs = np.zeros((len(ldca),4))
    LDC_coeffs[:,0] = ldca
    LDC_coeffs[:,1] = ldcb
    LDC_coeffs[:,2] = rprs
    LDC_coeffs[:,3] = kepler_snr
    
    filename2 = ldc_ratio_outfile[:-4]+f'_all.npy'
    np.save(filename2,LDC_coeffs)


    df_new =koi_conf_plans_tabl[koi_conf_plans_tabl['koi_model_snr']>= snr_cut]
    print(f'Number of planets with SNR >= {snr_cut} is {len(df_new)}')
    kepler_lcs_rprs = df_new["koi_ror"].to_numpy()
    kepler_lcs_ldca = df_new["koi_ldm_coeff1"].to_numpy()
    kepler_lcs_ldcb = df_new["koi_ldm_coeff2"].to_numpy()
    kepler_snr = df_new["koi_model_snr"].to_numpy()
    
    ldca, ldcb, rprs,mask = remove_nan_from_arrays(kepler_lcs_ldca, kepler_lcs_ldcb, kepler_lcs_rprs)
    kepler_snr = kepler_snr[mask]
    #print('kepler_snr',kepler_snr)
    print('Number of valid values in Kepler KOI for LDC and Rp/Rs',len(ldca))
    #np.isnan(a).any(),np.isnan(a).sum()
    LDC_coeffs = np.zeros((len(ldca),4))
    LDC_coeffs[:,0] = ldca
    LDC_coeffs[:,1] = ldcb
    LDC_coeffs[:,2] = rprs
    LDC_coeffs[:,3] = kepler_snr
    
    np.save(ldc_ratio_outfile,LDC_coeffs)

    return koi_conf_plans_tabl, filename2



def add_inverse_bins_as_input(df, bins, column='rprs',
                              outfile='binned_table.csv'):
    """
    Bin using input bins defined in 1/rprs space.

    Parameters
    ----------
    bins : list of [left, right]
        Bin edges in inverse-rprs space (1/rprs)
    """

    df = df.copy()

    rprs = df[column].to_numpy()
    #inv_rprs = 1.0 / rprs
    inv_rprs = np.round(1.0 / rprs).astype(int)
    #print('inv_rprs',inv_rprs)
    bins = np.asarray(bins)

    bin_idx = np.full(len(df), -1, dtype=int)

    # assign bins in inverse space
    for i, (l, r) in enumerate(bins):
        mask = (inv_rprs >= l) & (inv_rprs < r)
        bin_idx[mask] = i

    # include right edge for last bin
    l, r = bins[-1]
    bin_idx[(inv_rprs >= l) & (inv_rprs <= r)] = len(bins) - 1
    # print('bins',bins)
    # print('l, r(last bin)',l, r)
    # print('bin_idx[(inv_rprs >= l) & (inv_rprs <= r)]',inv_rprs[(inv_rprs >= l) & (inv_rprs <= r)])
    
    df['invrprs_bin_index'] = bin_idx

    # store inverse-space bins (as given)
    df['invrprs_bin_edges'] = [
        tuple(bins[i]) if i >= 0 else (-np.inf, np.inf)
        for i in bin_idx
    ]

    # convert back to rprs-space bins
    df['rprs_bin_edges'] = [
        (1.0 / bins[i][1], 1.0 / bins[i][0])
        if i >= 0 else (-p.inf, np.inf)
        for i in bin_idx
    ]

    df.to_csv(outfile, index=False)

    return df

        
def main(snr_cut = 50, n_rsrp_bins=15):
    obj = MLPreProcessing()
    figure_dir = Path(Base_Dir + "Figures/Kepler")
    figure_dir.mkdir(parents=True, exist_ok=True)
    
    ldc_ratio_outfile_1 =f"{obj.koi_table_folder}kepler_ldc_coeffs_conf_planets.npy"
    
    koi_conf_plans_tabl, ldc_ratio_outfile_2 = save_kepler_ldc_ratio(koi_table_folder =
                                                obj.koi_table_folder,
                          koi_table_filename = obj.koi_table_filename,
                          ldc_ratio_outfile = ldc_ratio_outfile_1,
                          snr_cut = snr_cut
                         )
    
    ldcs_coeffs_1 = np.load(ldc_ratio_outfile_2) # all sample
    ldcs_coeffs_2 = np.load(ldc_ratio_outfile_1) # subsample
    
    rp_rs_1 = ldcs_coeffs_1[:,2] # all planets
    rp_rs_2 = ldcs_coeffs_2[:,2] # planets with snr>snr_cut

    snr_1 = ldcs_coeffs_1[:,3] # all planets
    snr_2 = ldcs_coeffs_2[:,3] # planets with snr>snr_cut
    #print('snr_2',snr_2)

    # bin log(rprs) into n_rsrp_bins 
    hist = LogBinHistogram(nbins=n_rsrp_bins,figure_dir= figure_dir)
    hist.add(
        rp_rs_1,
        label="Kepler Planets",
        color='blue',
        alpha=0.4,
        snr = snr_1
    )
    inverse_bins_org, counts,snr_ranges = hist.add(
        rp_rs_2,
        label=f"Kepler Planets (SNR > {snr_cut}) ",
        color='red',
        alpha=0.4,
        snr = snr_2
    )    
    hist.show()
    print('Rs/Rp bins are',inverse_bins_org)
    print('Number of targets in each bin',counts)

    # sort the bins in increasing number of objects per bin
    # note: some bins may not have any real data in that bin
    mask = np.argsort(counts)[::-1] #(counts!=0)
    counts =counts[mask]
    inverse_bins = inverse_bins_org[mask]
    snr_ranges = snr_ranges[mask]

    rsrp_bins = np.zeros((len(counts),6))
    rsrp_bins[:,0:2] = inverse_bins
    rsrp_bins[:,2] = counts
    rsrp_bins[:,3:6] = snr_ranges
    np.save(obj.koi_table_folder+f'koi_rsrp_bin_info_snr{snr_cut}.csv',rsrp_bins)

    array = rsrp_bins
    # Save the table as png
    bins_rsrp = [f"[{row[0]}, {row[1]}]" for row in array]
    bins_snr = [f"[{row[3]}, {row[4]}, {row[5]}]" for row in array]
    # Create DataFrame
    df = pd.DataFrame({
        "Kepler Rs/Rp Bins": bins_rsrp,
        "No. of targets": array[:, 2],
        "SNR Ranges": bins_snr
    })
    
    print(df)
    
    # ---- Save table as PNG ----
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.axis('off')
    
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.savefig(figure_dir/"kepler_radius_bins_table.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved as {figure_dir}/kepler_radius_bins_table.png")

    
    # create new table containing extra columns with bin index and bin ranges added
    table_subset = koi_conf_plans_tabl[koi_conf_plans_tabl['koi_model_snr']>= snr_cut]
    table_subset_new_name = obj.koi_table_folder + f'koi_cumulative_snr{snr_cut}.csv'
    add_inverse_bins_as_input(table_subset, inverse_bins_org, column='koi_ror',
                                  outfile= table_subset_new_name)
    
    table_subset_new = pd.read_csv(table_subset_new_name,comment='#')
    return table_subset_new
        
if __name__ == "__main__":
    obj = MLPreProcessing()
    
