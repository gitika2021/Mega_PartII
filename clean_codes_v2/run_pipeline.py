import numpy as np
import sys
from class_modules import MLPreProcessing
import train_on_kepler_noise

if __name__ == "__main__":
    # train = bool(sys.argv[1])
    # Num = int(sys.argv[2])
    # N = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    # maps_path = sys.argv[4] if len(sys.argv) > 4 else None
    # nproc = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    # # train_frac and seed should also be in user control

    # rsrp1 = int(sys.argv[4])
    # rsrp2 = int(sys.argv[5])
    # koi_table_folder = int(sys.argv[6])
    # koi_table_filename = int(sys.argv[7])

    train = True
    Num = 10
    N = 1
    maps_path = None
    nproc = 4
    rsrp1 = 5
    rsrp2 = 10
    train_frac = 0.8
    seed = None

    # only needed for actual training
    epochs = 3
    batch_size = 10 # 32
    n_scale = 2
    device = 'cpu' # "cuda" if torch.cuda.is_available() else "cpu"
    resume = False
    checkpoint_freq = 3
    
    obj = MLPreProcessing(Num=Num,N=N,maps_path=maps_path,nproc=nproc,rsrp1=rsrp1,rsrp2=rsrp2,train_frac=train_frac,seed=seed)
    if train:
        train_on_kepler_noise.main(obj.train_dir,obj.model_dir,
                                   epochs,batch_size,n_scale,device,resume,checkpoint_freq,figpath=str(obj.figure_dir))
    else:
        obj.execute()
    
