import subprocess
import shape_utils,gen_ldc_ratio_grid,genlc_with_grid
import add_noise_to_lcs_files
from pathlib import Path
from paths import *

class MLTraining():
    def __init__(self,Num=1000,N=1,maps_path=None, rsrp1=5, rsrp2=10,nproc=32):
        self.Num = Num
        self.N = N
        # base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
        base_dir = Path(Base_Dir) / "Data" # this is actually data directory
        
        base_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir = base_dir
        self.maps_path = maps_path
        
        self.figure_path = self.base_dir / "figures"
        self.figure_path.mkdir(parents=True, exist_ok=True)

        # shapes directory
        self.shape_dir = self.base_dir / "OM10"
        self.shape_dir.mkdir(parents=True, exist_ok=True)
        
        self.koi_table_folder = Kepler_Dir
        self.koi_table_filename = KOI_Table_Filename
        self.kepler_error_file = self.koi_table_folder + Kepler_Error_Filename

        self.rsrp1 = rsrp1
        self.rsrp2 = rsrp2
        
        self.out_dir_lc = self.base_dir / "LC10"
        self.out_dir_lc.mkdir(parents=True, exist_ok=True)
        
        self.out_dir_proc_lc = self.out_dir_lc / "proc/RsRp_{0:d}_{1:d}".format(self.rsrp1,self.rsrp2)
        self.out_dir_proc_lc.mkdir(parents=True, exist_ok=True)
        self.out_stem_lc = self.out_dir_proc_lc / f"{self.N}"
        self.out_file_lc = str(self.out_stem_lc) + 'LC.npy' # this is now a string
    
        self.out_dir_orig_lc = self.out_dir_lc / "orig/RsRp_{0:d}_{1:d}".format(self.rsrp1,self.rsrp2)
        self.out_dir_orig_lc.mkdir(parents=True, exist_ok=True)

        self.ldc_ratio_grid_file = gen_ldc_ratio_grid.main(rsrp1=self.rsrp1,rsrp2=self.rsrp2,
                                                           koi_table_folder=self.koi_table_folder,
                                                           koi_table_filename=self.koi_table_filename,base_dir=self.base_dir)
        self.nproc = nproc

        print("... initialization complete.")

    def gen_shapes(self):
        shape_utils.main(Num=self.Num,N=self.N,shape_dir=self.shape_dir,maps_path=self.maps_path) 
        # check if files created
        return
    
                
    def gen_ltcrvs(self):        
        genlc_with_grid.main(self.N,1,self.base_dir,self.shape_dir,self.ldc_ratio_grid_file,self.out_stem_lc,self.out_dir_orig_lc,
                             nproc=self.nproc)
        return


    def add_noise(self):
        add_noise_to_lcs_files.main(self.out_file_lc,self.kepler_error_file,self.figure_path)
        return
    

    def execute(self):
        #########################
        # change this block to allow multiple batches of light curves
        if not (self.shape_dir / f"{self.N}.npy").is_file():
            self.gen_shapes()
            
        if not (self.out_dir_proc_lc / f"{self.N}LC.npy").is_file():
            self.gen_ltcrvs()

        if Path(self.out_file_lc).is_file():
            self.add_noise()
            
        return

        
        
