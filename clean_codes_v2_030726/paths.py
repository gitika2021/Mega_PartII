from pathlib import Path
'''

# IITT Workstation Paths
Home = '/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Mega_PartII/'
Base_Dir = Home+'Test/'
Kepler_Dir = Home+'Kepler/'
Kepler_LCS_Dir = Home+'Kepler_Binned_LCS/'
Config_Dir = Home+'Config/'
KOI_Table_Filename = 'koi_cumulative_2025.06.28_01.24.15.csv'
Kepler_Error_Filename = 'kepler_folded_lcs_snr50_all_binned_err.npy'
Infer_LC_Dir = Base_Dir+"Kepler_RsRp_Bins/"
'''

# Pegasus Paths
#User = '/mnt/home/project/cshukla.gitika/'
User_Dir = str(Path.home()) + '/'
User_Sub_Dir = 'Gitika/' + 'Github_Repositories/'
Working_Dir = User_Dir + User_Sub_Dir

#Working_Dir = Path(Working_Dir)
#Working_Dir.mkdir(parents=True, exist_ok=True)

Home = Working_Dir + 'Mega_PartII/'
#Base_Dir = Home+'Test_Runs_Pegasus/'
Base_Dir = Home+'Pipeline_Runs/'
Kepler_Dir = Home+'Kepler/'
Kepler_LCS_Dir = Home+'Kepler_Binned_LCS/'
Config_Dir = Home+'Config/'
KOI_Table_Filename = 'koi_cumulative_2025.06.28_01.24.15.csv'
Kepler_Error_Filename = 'kepler_folded_lcs_snr50_all_binned_err.npy'
#Infer_LC_Dir = Home+"Raw_LC/"
Infer_LC_Dir = Base_Dir+"Kepler_RsRp_Bins/"

