import numpy as np
import sys
from class_modules import MLPreProcessing, MLInference
import train_on_kepler_noise
from paths import Config_Dir, Infer_LC_Dir
import argparse,json
from rich import print as prcolor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lightcurve generation and model training.")
    parser.add_argument("--config-file",type=str,required=True, help="name of config file (e.g., example_config.json)")
    parser.add_argument("--train",type=int,help="whether to implement training (True) or pre-processing (False).")
    parser.add_argument("--N",type=int,help="index number for pre-processing batch.")
    parser.add_argument("--Num",type=int,help="number of shapes to generate")
    parser.add_argument("--test",type=int,help="whether to implement testing on simulated data or real data")
    parser.add_argument("--fresh_run",type=int,help="whether to implement do full fresh run (eg. generating shapes, ltcrvs etc.) ")

    args = parser.parse_args()

    config_file = args.config_file
    with open(Config_Dir+config_file,'r') as f:
        config = json.load(f)
    
    train = bool(args.train) if args.train is not None else config['train']
    N = args.N if args.N is not None else config['N']
    Num = args.Num if args.Num is not None else config.get('Num',10)
    test = bool(args.test) if args.test is not None else config['test']
    fresh_run = bool(args.fresh_run) if args.fresh_run is not None else config['fresh_run']
    
    #Num = config.get('Num',10)
    maps_path = config.get('maps_path',None)
    nproc = config.get('nproc',4)
    rsrp1 = config.get('rsrp1',5)
    rsrp2 = config.get('rsrp2',10)
    train_frac = config.get('train_frac',0.8)
    seed = config.get('seed',None) + N
    maps_folder_str = config.get('maps_folder_str',10)
    n_scale = config.get('n_scale',2)
    
    # obj = MLPreProcessing(Num=Num,N=N,maps_path=maps_path,nproc=nproc,rsrp1=rsrp1,rsrp2=rsrp2,train_frac=train_frac,seed=seed, maps_folder_str = maps_folder_str, test=test, fresh_run = fresh_run)
        
    if test is True:
        prcolor("[bold green]Note: Test shapes are always saved as 99.npy")
        N = 99
        #Num = Num+21
        obj = MLPreProcessing(Num=Num,N=N,maps_path=maps_path,nproc=nproc,rsrp1=rsrp1,rsrp2=rsrp2,train_frac=train_frac,seed=seed, maps_folder_str = maps_folder_str, test=test, fresh_run = fresh_run)
        
        obj.execute()
        
        print("Infering shape of simulated test LCs")
        infer = MLInference(lc_dir=obj.noisy_ltcrv_folder,nproc=nproc, rsrp1=rsrp1, rsrp2=rsrp2,
                     n_scale=n_scale, N=N, obj = obj)
        infer.execute()
        #infer.plot_prediction_orig_maps()
        

    elif test is False:
        obj = MLPreProcessing(Num=Num,N=N,maps_path=maps_path,nproc=nproc,rsrp1=rsrp1,rsrp2=rsrp2,train_frac=train_frac,seed=seed, maps_folder_str = maps_folder_str, test=test, fresh_run = fresh_run)
        
        print("Infering shape from LC using trained model for real data")
        infer = MLInference(lc_dir=Infer_LC_Dir,nproc=nproc, rsrp1=rsrp1, rsrp2=rsrp2,
                     n_scale=n_scale, N=None, obj = obj)
        infer.execute()      
    
