import subprocess
import shape_utils 
import gen_ldc_ratio_grid
from pathlib import Path

class MLInference():
    def __init__(self,base_dir=None ):
        base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
        base_dir = base_dir / "Data"
        
        base_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir = base_dir
        


    def gen_shapes(self, Num, N, base_dir, maps_path):
        shape_utils.main(Num, N = N,base_dir = self.base_dir,maps_path=maps_path)    

    def gen_ldc_grid(self, rsrp1='', rsrp2='',koi_table_folder=None, koi_table_filename=None, base_dir=None):
        gen_ldc_ratio_grid.main(rsrp1=rsrp1, rsrp2=rsrp2, koi_table_folder =koi_table_folder, 
                                koi_table_filename=koi_table_filename, base_dir = self.base_dir)    

    def gen_ltcrvs(self, Num, N, base_dir, maps_path):
        shape_utils.main(Num, N = N,base_dir = self.base_dir,maps_path=maps_path)    
