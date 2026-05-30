import numpy as np
import matplotlib.pyplot as plt
from lightkurve import LightCurve
from class_modules import MLPreProcessing
from paths import *
from koi_table import KoiTableObjs as koitab

class GenPlots_MegaPartII():
        def __init__(self, rsrp1=10,rsrp2=12, snr=1000, kepobj_file = "kplr007382313_kepler_1220b_binned.npz"):
            self.mlprep = MLPreProcessing()
            self.kepobj_file = kepobj_file
            self.rsrp1 = rsrp1
            self.rsrp2 = rsrp2
            self.snr = snr
            
            
            pass

        def get_kepler_target(self):
            koi = koitab(files_dir = Kepler_Dir,koi_file_name = KOI_Table_Filename,verbose=False)
            koi_table = koi.koi_table
            #print(koi_table)
            koi.get_koi_confirmed()
            koi_conf_plans_tabl = koi.koi_conf_plans_tab
            plans_table = koi_conf_plans_tabl
            plans_table_sel = plans_table[(plans_table['koi_model_snr'] >= self.snr) &
                                (plans_table['koi_ror'] > 1/self.rsrp2) &
                                (plans_table['koi_ror'] < 1/self.rsrp1)]
            plans_table_sel2 = plans_table_sel[plans_table_sel['koi_model_snr']==np.max(plans_table_sel['koi_model_snr'])]
            print('plans_table_sel2',plans_table_sel2['koi_model_snr'])
            self.kepobj = plans_table_sel2['kepname']
            self.kepid_str = plans_table_sel2['kepid_str']
            kep_filename = self.kepid_str + '_'+ 'k' + self.kepobj[1:6] + '_' + self.kepobj[7:]+'_binned.npz'
            print(koi_conf_plans_tabl.columns,plans_table_sel2['kepid_str'])
            print('kep_filename',kep_filename)
        def overlays_lcs_circle(self):
            data = np.load(file_path)
            lc_fold_load = LightCurve(time=data['time'], flux=data['flux'], flux_err=data['flux_err'])
            x = lc_fold_load.time.value
            y = lc_fold_load.flux.value
            yerr = lc_fold_load.flux_err.value

            
            
            
            