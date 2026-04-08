
import sys
sys.path.append("/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/pyscripts")
sys.path.append("/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git")
sys.path.append("/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Python-Scripts")


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import pandas as pd
from plotting_utils import *

# my modules
from koi_table import KoiTableObjs as koitab
from lightkurve_singlev2 import LightkurveAnalysisSingleObjV2
from lightkurve_batchv2 import LcDownloadBatchInParallelV2
from plotting_utils import *
from noise_utils_kepler import *
from scipy.stats import gaussian_kde

       
def save_kepler_ldc_ratio(koi_table_file = "/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Kepler/",
                         ldc_ratio_outfile = "/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Kepler/kepler_ldc_coeffs_conf_planets.npy"):
    """
    kepler koi table file: koi_cumulative_2025.06.28_01.24.15.csv should be downloaded first and saved in files_dir
    ldc_ratio_outfile: (N,a,b,rp/rs) from kepler koi table after removing nans etc.
    """
    koi = koitab(files_dir = koi_table_file,verbose=False)
    koi_table = koi.koi_table
    #print(koi_table)
    koi.get_koi_confirmed()
    koi_conf_plans_tabl = koi.koi_conf_plans_tab
    
    df_new =koi_conf_plans_tabl # koi_table # 
    kepler_lcs_rprs = df_new["koi_ror"].to_numpy()
    kepler_lcs_ldca = df_new["koi_ldm_coeff1"].to_numpy()
    kepler_lcs_ldcb = df_new["koi_ldm_coeff2"].to_numpy()
    
    ldca, ldcb, rprs = remove_nan_from_arrays(kepler_lcs_ldca, kepler_lcs_ldcb, kepler_lcs_rprs)
    print('Number of valid values in Kepler KOI for LDC and Rp/Rs',len(ldca))
    #np.isnan(a).any(),np.isnan(a).sum()
    LDC_coeffs = np.zeros((len(ldca),3))
    LDC_coeffs[:,0] = ldca
    LDC_coeffs[:,1] = ldcb
    LDC_coeffs[:,2] = rprs
    
    np.save(ldc_ratio_outfile,LDC_coeffs)
    return ldc_ratio_outfile

def generate_band(a, b, method="constant", value=1.0, size=2000, show=False):
    """
    Generate upper and lower bands for correlated arrays a and b.
    Here a and b are Limb Darkeining Coefficients a and b
    
    Parameters
    ----------
    a : array-like
        X values
    b : array-like
        Y values
    method : str
        "constant"      -> fixed vertical shift
        "proportional"  -> percentage band
        "std"           -> based on residual std from linear fit
    value : float
        Shift amount:
            - constant: absolute shift
            - proportional: percentage (0.05 = 5%)
            - std: multiplier of residual std (e.g., 2 for 2σ)

    Returns
    -------
    a, b_upper, b_lower
    """

    a = np.asarray(a)
    b = np.asarray(b)

    if method == "constant":
        delta = value
        b_upper = b + delta
        b_lower = b - delta

    elif method == "proportional":
        p = value
        b_upper = b * (1 + p)
        b_lower = b * (1 - p)

    elif method == "std":
        # Linear regression fit
        coeffs = np.polyfit(a, b, 1)
        b_fit = np.polyval(coeffs, a)
        std_a = np.std(a)
        
        residuals = b - b_fit
        std = np.std(residuals)
        if show:
            plt.scatter(a,b,color='k')
            plt.plot(a,b_fit,color='r')
            #plt.scatter(a+5*std_a,np.polyval(coeffs, a+5*std_a), color='r',s=1)
            plt.plot(a,b_fit+value * std,'--r')
            plt.plot(a,b_fit-value * std,'--r')
            # plt.plot(a,b_fit+5*std,color='r')
            plt.show()
        
        # delta = value * std
        # b_upper = b + delta
        # b_lower = b - delta

        delta = value * std
        b_upper = b_fit + delta
        b_lower = b_fit - delta

        fn_up = interp1d(a, b_upper, kind='linear')
        fn_lo = interp1d(a, b_lower, kind='linear')
        a = np.linspace(a.min(),a.max(),size)
        b_upper = fn_up(a)
        b_lower = fn_lo(a)
    else:
        raise ValueError("method must be 'constant', 'proportional', or 'std'")

    
    return a, b_upper, b_lower, std, b_fit, coeffs

def check_negative_intensity(u1, u2, n_mu=1000):
    """
    Check if quadratic limb darkening produces negative intensity.

    Parameters
    ----------
    u1 : array-like
    u2 : array-like
    n_mu : int
        number of mu grid points

    Returns
    -------
    negative_mask : boolean array
        True where intensity becomes negative
    """

    u1 = np.asarray(u1)
    u2 = np.asarray(u2)

    mu = np.linspace(0, 1, n_mu)

    negative_mask = np.zeros(len(u1), dtype=bool)

    for i in range(len(u1)):
        I = 1 - u1[i]*(1 - mu) - u2[i]*(1 - mu)**2
        if np.any(I < 0):
            negative_mask[i] = True

    return negative_mask
    

def generate_uniform_grid_ldc_ratio(sample_size = 500000, rng = np.random.default_rng(20),
                      ldca_min=0.2, ldca_max=0.7, ldcb_min=0.06, ldcb_max=0.5,
                      rprs_min = 0.10, rprs_max = 0.5,
                      savefile = "/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/LDC_RPRS/ldc_ratio_grid.npy"):
    ldca = np.random.uniform(ldca_min, ldca_max, size=sample_size)
    ldcb = np.random.uniform(ldcb_min, ldcb_max, size=sample_size)
    rprs = np.random.uniform(rprs_min, rprs_max, size=sample_size)

    coeff_arr = np.zeros((sample_size,3))
    coeff_arr[:,0] = ldca
    coeff_arr[:,1] = ldcb
    coeff_arr[:,2] = rprs
    np.save(savefile,coeff_arr)
    return savefile


def remove_nan_from_arrays(a, b, c):
    """
    Remove entries where any of the arrays contains NaN.
    Keeps indices aligned across arrays.
    """
    
    mask = ~(np.isnan(a) | np.isnan(b) | np.isnan(c))
    
    a_clean = a[mask]
    b_clean = b[mask]
    c_clean = c[mask]
    
    return a_clean, b_clean, c_clean

def kde_sampling(data, n_samples=7745, low_cut=90, hig_cut=98, percentile_flag=True):

    
    if percentile_flag == False:
        low = low_cut
        high = hig_cut
        
    elif percentile_flag == True:
        low = np.percentile(data, low_cut)
        high = np.percentile(data, hig_cut)
        
    data_cut = data[(data >= low) & (data <= high)]

    kde = gaussian_kde(data_cut, bw_method=0.05)

    samples = np.array([])

    while samples.size < n_samples:
        new = kde.resample(n_samples)[0]
        new = new[(new >= low) & (new <= high)]
        samples = np.concatenate((samples, new))

    return samples[:n_samples]

# --- Main Runner ---

def run_ldc_ratio_generator(loc_cut_rprs=0.2, high_cut_rprs=0.5, sampling = 'kde', outfile="/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/LDC_RPRS/ldc_ratio_grid_set.npy"):
    """
    sampling = 'kde'/'uni'
    kde: choose when want to generate same distribution as Kepler
    loc_cut_rprs, high_cut_rprs = float : generates uniform numbers between this
    """
    planet_ldc_file = save_kepler_ldc_ratio()
    ldcs_coeffs = np.load(planet_ldc_file)
    
    a = ldcs_coeffs[:,0] #kepler_lcs_ldca
    b = ldcs_coeffs[:,1] #kepler_lcs_ldcb
    rprs = ldcs_coeffs[:,2]
    print('a.min(),a.max()',a.min(),a.max())
    print('b.min(),b.max()',b.min(),b.max())
    print('rprs.min(),rprs.max()',rprs.min(),rprs.max())
    
    low_perc = 15
    hig_perc = 100
    # loc_cut_rprs = np.round(np.percentile(rprs,low_perc),3)
    # high_cut_rprs = np.round(np.percentile(rprs,hig_perc),3)
    
    # loc_cut_rprs = 1/5.0#0.083 
    # high_cut_rprs = 1/2.0#0.125
    print(f"loc_cut_rprs ={loc_cut_rprs}, high_cut_rprs = {high_cut_rprs}")
    
    rng = np.random.default_rng(50)
    values = rng.uniform(0, 6, 20) 
    #values= np.arange(0,5.0,0.1) # for more than 14.8 intensity becomes negative non-physical
    
    #anew, upper, lower,std,b_fit, coeffs = generate_band(a, b, method="std", value=0, show=True)
    # a_sel = np.array([0.2,0.4,0.6,0.8])
    # b_sel = np.polyval(coeffs, a_sel)
    # print('b_sel',b_sel)
    # anew, upper1, lower1,std,b_fit = generate_band(a, b, method="std", value=4, show=True)

    ## N= this is like being 0 to 6 sigma away from reference line LDC and generating 20 such lines
    ## M = Total number of points on each line 
    ## 2*M = Taking say 2sigma above and 2 sigma below line
    
    size= 125
    N = len(values)
    M = size#len(a)
    pairs = np.zeros((N*2*M+len(a),4), dtype=float)
    print('pairs.shape',pairs.shape)

    if sampling == 'uni':
        #rprs_grid = kde_sampling(rprs, n_samples=pairs.shape[0], low_cut_perc=low_perc, hig_cut_perc=hig_perc)
        rprs_grid = kde_sampling(rprs, n_samples=pairs.shape[0], low_cut=loc_cut_rprs, hig_cut= high_cut_rprs,percentile_flag = False)
        
        print('rprs_grid',rprs_grid.shape,pairs.shape[0])
        for i, valuei in enumerate(values):
            anew, upper, lower,std, b_fit,coeffsnew = generate_band(a, b, method="std", value=valuei, size= size)
            aext = np.concatenate((anew,anew))
            bext = np.concatenate((upper,lower))
            ni = len(aext)
            rng = np.random.default_rng(20)
            rprsi = rng.uniform(loc_cut_rprs, high_cut_rprs, size=len(aext))
        
            pairs[i*ni:(i+1)*ni,0] = aext
            pairs[i*ni:(i+1)*ni,1] = bext
            pairs[i*ni:(i+1)*ni,2] = rprsi
            pairs[i*ni:(i+1)*ni,3] = np.round(1/rprsi,0).astype(int)
        #     plt.scatter(anew, upper, c='k', s=1, alpha=0.3)
        #     plt.scatter(anew, lower, c='k', s=1, alpha=0.3)
        # plt.show()
        
        pairs[N*2*M:,0] = a
        pairs[N*2*M:,1] = b
        # pairs[:,2] = rprs_grid
        # pairs[:,3] = np.round(1/rprs_grid,0)
        
        #pairs[:,2] = 1/pairs[:,3]
        pairs[N*2*M:,2] = rng.uniform(loc_cut_rprs, high_cut_rprs, size=len(rprs))#rprs
        pairs[N*2*M:,3] = np.round(1/pairs[N*2*M:,2],0).astype(int)#np.round(1/rprs,0)
        print(f"Number of LDC & Rp/Rs points is {pairs.shape[0]}")
        #outfile = "/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/LDC_RPRS/ldc_ratio_grid_set.npy"
        np.save(outfile, pairs)
        
        print('radius ratio (Rs/Rp)',pairs[:,3],pairs[:,3].min(),pairs[:,3].max())
        print(np.any(pairs[:,3] == 0))

    elif sampling == 'kde':
        rprs_grid = kde_sampling(rprs, n_samples=pairs.shape[0], low_cut=loc_cut_rprs, hig_cut= high_cut_rprs,percentile_flag = False)        
        print('rprs_grid',rprs_grid.shape,pairs.shape[0])
        for i, valuei in enumerate(values):
            anew, upper, lower,std, b_fit,coeffsnew = generate_band(a, b, method="std", value=valuei, size= size)
            aext = np.concatenate((anew,anew))
            bext = np.concatenate((upper,lower))
            ni = len(aext)
            rng = np.random.default_rng(20)
            #rprsi = rng.uniform(loc_cut_rprs, high_cut_rprs, size=len(aext))
        
            pairs[i*ni:(i+1)*ni,0] = aext
            pairs[i*ni:(i+1)*ni,1] = bext
            # pairs[i*ni:(i+1)*ni,2] = rprsi
            # pairs[i*ni:(i+1)*ni,3] = np.round(1/rprsi,0)
        #     plt.scatter(anew, upper, c='k', s=1, alpha=0.3)
        #     plt.scatter(anew, lower, c='k', s=1, alpha=0.3)
        # plt.show()
        
        pairs[N*2*M:,0] = a
        pairs[N*2*M:,1] = b
        pairs[:,2] = rprs_grid
        pairs[:,3] = np.round(1/rprs_grid,0).astype(int)
        
        #pairs[:,2] = 1/pairs[:,3]
        # pairs[N*2*M:,2] = rng.uniform(loc_cut_rprs, high_cut_rprs, size=len(rprs))#rprs
        # pairs[N*2*M:,3] = np.round(1/pairs[N*2*M:,2],0)#np.round(1/rprs,0)
        print(f"Number of LDC & Rp/Rs points is {pairs.shape[0]}")
        #outfile = "/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/LDC_RPRS/ldc_ratio_grid_set.npy"
        np.save(outfile, pairs)
        
        print('radius ratio (Rs/Rp)',pairs[:,3],pairs[:,3].min(),pairs[:,3].max())
        print(np.any(pairs[:,3] == 0))

def plot_grid(outfile="/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/LDC_RPRS/ldc_ratio_grid_set.npy",figure_path="/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/figures/ldc_rprs_grid.png"):
        planet_ldc_file = save_kepler_ldc_ratio()
        ldcs_coeffs = np.load(planet_ldc_file)
        
        ldca = ldcs_coeffs[:,0] #kepler_lcs_ldca
        ldcb = ldcs_coeffs[:,1] #kepler_lcs_ldcb
        rprs = ldcs_coeffs[:,2]

        train_meta = np.load(outfile)

        fontsize = 24
        # check distribution of median error
        fig, ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].set_xlabel("LDC coeff a")
        ax[0].set_ylabel("LDC coeff b")
        
        ax[1].set_xlabel(r"$R_p/R_s$")
        ax[1].set_ylabel("Counts")
        
        ax[0].scatter(ldca, ldcb, s=2, color='r', label = 'Kepler KOI', alpha=0.9, zorder=2)
        ax[0].scatter(train_meta[:,0], train_meta[:,1], s=2, color="gray", label = 'Generated Grid', alpha=0.3, zorder=1)
        
        #ax[1].hist(rprs_grid,label = 'Generated Grid', alpha=0.5, color="gray", edgecolor="black")
        
        ax[1].hist(rprs,label = 'Kepler KOI', alpha=0.5, color="red", edgecolor="black")
        ax[1].hist(train_meta[:,2],label = 'Generated Grid', alpha=0.5, color="gray", edgecolor="black")
        ax[1].set_yscale('log')
        #ax[1].set_xscale('log')
        # ax[2].hist(np.round(1/rprs,0),label = 'Kepler KOI', alpha=0.5, color="red", edgecolor="black")
        #ax[2].hist(np.round(1/rprs,0),label = 'Kepler KOI', alpha=0.5, color="red", edgecolor="black")
        # ax[2].hist(train_meta[:,3],label = 'Generated Grid', alpha=0.5, color="gray", edgecolor="black")
        # ax[2].set_yscale('log')
        
        ax[0].legend()
        ax[1].legend()
        plt.savefig(
            figure_path,
            dpi=500,
            bbox_inches='tight',
            pad_inches=0.2
        )
        plt.show()
    
if __name__ == "__main__":
    run_ldc_ratio_generator(sampling = 'uni')
    plot_grid()

    run_ldc_ratio_generator(loc_cut_rprs=0.05, high_cut_rprs=0.1, sampling = 'kde', outfile="/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/LDC_RPRS/ldc_ratio_grid_set_train1.npy")
    plot_grid(figure_path="/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/figures/ldc_rprs_grid_train1.png")

    outfile = "/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/LDC_RPRS/ldc_ratio_grid_set.npy"
    train_meta = np.load(outfile)
    
    a = train_meta[:,0]
    b = train_meta[:,1]
    negative_mask = check_negative_intensity(a, b, n_mu=1000)
    print("LDC giving negative intesnity", np.where(negative_mask==True))
    # a_neg = a[np.where(negative_mask==True)]
    # b_neg = b[np.where(negative_mask==True)]