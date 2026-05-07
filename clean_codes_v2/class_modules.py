import subprocess
import shape_utils 
import gen_ldc_ratio_grid
from pathlib import Path
from paths import *

class MLTraining():
    def __init__(self,Num=1000,N=1,maps_path=None, rsrp1=5, rsrp2=10):
        self.Num = Num
        self.N = N
        # base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
        base_dir = Path(Base_Dir) / "Data" # this is actually data directory
        
        base_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir = base_dir
        self.maps_path = maps_path

        self.koi_table_folder = Kepler_Dir
        self.koi_table_filename = KOI_Table_Filename

        self.rsrp1 = rsrp1
        self.rsrp2 = rsrp2        

        self.ldc_ratio_grid_file = gen_ldc_ratio_grid.main(rsrp1=self.rsrp1, 
                                                           rsrp2=self.rsrp2,
                                                           koi_table_folder=self.koi_table_folder,koi_table_filename=self.koi_table_filename,base_dir=self.base_dir)

    def gen_shapes(self):
        self.om10_dir = shape_utils.main(Num=self.Num,N=self.N,base_dir=self.base_dir,
                                    maps_path=self.maps_path) 
        # check if files created

        if out_dir_lc is None:
            out_dir_lc = base_dir / "LC10"
        else:
            out_dir_lc = Path(out_dir_lc)
        out_dir_lc.mkdir(parents=True, exist_ok=True)
    
        out_dir_orig_lc = out_dir_lc / "orig"
        out_dir_orig_lc.mkdir(parents=True, exist_ok=True)
        
        out_dir_proc_lc = out_dir_lc / "proc"
        out_dir_proc_lc.mkdir(parents=True, exist_ok=True)
    
                

    def gen_ltcrvs(self):
        genlc_with_grid.main(N,
    n,
    #rsrp=10,
    base_dir,
    shape_dir,
    #shape_file=None,
    ldc_ratio_path,
    out_dir_lc=None,
    rsrp_low = None,
    rsrp_high = None,
)
        pass


    def execute(self):
        #########################
        # change this block to allow multiple batches of light curves
        self.gen_shapes()
        ##########################


        
        
