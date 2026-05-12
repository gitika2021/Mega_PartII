import numpy as np
import sys
from class_modules import MLInference
import train_on_kepler_noise
from paths import Config_Dir, infer_lc_dir
import argparse,json

if __name__ == "__main__":
    nproc = 4
    obj = MLInference(maps_dir=infer_lc_dir,nproc=None)
    obj.execute()