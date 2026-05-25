from class_modules import SaveKeplerToRsRpBins
from paths import *
if __name__ == "__main__":
    #kepler_distribution_utils.main(snr_cut = 50, n_rsrp_bins=21)
    lcs_dir = Home+'Kepler_Binned_LCS/'
    
    kepobj = SaveKeplerToRsRpBins(lcs_dir, snr_cut = 50, n_rsrp_bins=21)
    kepobj.distribute_kepler_into_rsrp_bins()