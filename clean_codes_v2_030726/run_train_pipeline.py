import numpy as np
import sys
from class_modules import MLPreProcessing
import train_on_kepler_noise
from paths import Config_Dir
import argparse,json
from rich import print as prcolor

if __name__ == "__main__":
    # train = True
    # Num = 10
    # N = 1
    # maps_path = None
    # nproc = 4
    # rsrp1 = 5
    # rsrp2 = 10
    # train_frac = 0.8
    # seed = None

    # # only needed for actual training
    # epochs = 3
    # batch_size = 10 # 32
    # n_scale = 2
    # device = 'cpu' # "cuda" if torch.cuda.is_available() else "cpu"
    # resume = False
    # checkpoint_freq = 3

    parser = argparse.ArgumentParser(description="Lightcurve generation and model training.")
    parser.add_argument("--config-file",type=str,required=True, help="name of config file (e.g., example_config.json)")
    parser.add_argument("--train",type=int,help="whether to implement training (True) or pre-processing (False).")
    parser.add_argument("--N",type=int,help="index number for pre-processing batch.")
    parser.add_argument("--Num",type=int,help="number of shapes to generate")
    parser.add_argument("--test",type=int,help="whether to implement testing on simulated data or real data")
    parser.add_argument("--fresh_run",type=int,help="whether to implement do full fresh run (eg. generating shapes, ltcrvs etc.) ")

    args = parser.parse_args()
    
    config_file = args.config_file
    print('Config file used',Config_Dir+config_file)
    with open(Config_Dir+config_file,'r') as f:
        config = json.load(f)
    
    train = bool(args.train) if args.train is not None else config['train']
    N = args.N if args.N is not None else config['N']
    Num = args.Num if args.Num is not None else config.get('Num',10)
    test = False #bool(args.test) if args.test is not None else config['test']
    #fresh_run = bool(args.fresh_run) if args.fresh_run is not None else config['fresh_run']
    if args.fresh_run != 2:
        fresh_run = bool(args.fresh_run) if args.fresh_run is not None else config['fresh_run']
    else:
        fresh_run = args.fresh_run if args.fresh_run is not None else config['fresh_run']
        
    N = 1 if N==99 else N
    prcolor(f"[bold green]Updated filename to {N}")
    
    #Num = config.get('Num',10)
    maps_path = config.get('maps_path',None)
    nproc = config.get('nproc',4)
    rsrp1 = config.get('rsrp1',5)
    rsrp2 = config.get('rsrp2',10)
    train_frac = config.get('train_frac',0.8)
    seed = config.get('seed',None) + N
    maps_folder_str = config.get('maps_folder_str',10)
    n_scale = config.get('n_scale',2)
    snr_min = config.get('snr_min',100)
    snr_max = config.get('snr_max',500)
    noise = config.get('noise','real')
    train_network = config.get('train_network','fixed_noise')
    print(f"SNR Range: [{snr_min}:{snr_max}], noise: {noise}, train_network: {train_network}")
    obj = MLPreProcessing(Num=Num,N=N,maps_path=maps_path,nproc=nproc,rsrp1=rsrp1,rsrp2=rsrp2,train_frac=train_frac,seed=seed, maps_folder_str = maps_folder_str, test=test, fresh_run = fresh_run, snr_min=snr_min,snr_max=snr_max,noise=noise)
    if train:
        import torch
        epochs = config.get('epochs',3)
        batch_size = config.get('batch_size',10)
        n_scale = config.get('n_scale',2)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        resume = config.get('resume',False)
        checkpoint_freq = config.get('checkpoint_freq',3)
        print(f"resume",resume)
        print(f"For training device used is: {device}")
        train_on_kepler_noise.main(obj.train_dir,obj.model_dir,
                                   epochs,batch_size,n_scale,device,resume,checkpoint_freq,figpath=str(obj.figure_dir),
                                  train_network=train_network)
        
    elif fresh_run==2:
        print("Generating shapes only")
        obj.gen_shapes()
    elif fresh_run==True or fresh_run==False:
        print("Running full excute pipeline")
        obj.execute()

    
