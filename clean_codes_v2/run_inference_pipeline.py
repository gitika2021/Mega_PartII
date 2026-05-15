import numpy as np
import sys
from class_modules import MLInference
import train_on_kepler_noise
from paths import Config_Dir, Infer_LC_Dir
import argparse,json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lightcurve generation and model training.")
    parser.add_argument("--config-file",type=str,required=True, help="name of config file (e.g., example_config.json)")

    args = parser.parse_args()

    config_file = args.config_file
    with open(Config_Dir+config_file,'r') as f:
        config = json.load(f)
        
    n_scale = config.get('n_scale',2)
    rsrp1 = config.get('rsrp1',5)
    rsrp2 = config.get('rsrp2',10)
    
    nproc = 4
    obj = MLInference(lc_dir=Infer_LC_Dir,nproc=nproc, rsrp1=rsrp1, rsrp2=rsrp2,
                     n_scale=n_scale)
    obj.execute()