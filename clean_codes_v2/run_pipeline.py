import sys
from class_modules import MLTraining


if __name__ == "__main__":
    Num = int(sys.argv[1])
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    base_dir = sys.argv[3] if len(sys.argv) > 3 else None
    maps_path = sys.argv[4] if len(sys.argv) > 4 else None

    # rsrp1 = int(sys.argv[5])
    # rsrp2 = int(sys.argv[6])
    # koi_table_folder = int(sys.argv[7])
    # koi_table_filename = int(sys.argv[8])
    
    obj = MLTraining(Num=Num,N=N,base_dir=base_dir,maps_path=maps_path)

    # generate "Num" random bezier shapes
    # obj.gen_shapes(Num, N=N,base_dir=base_dir, maps_path = maps_path)
    # outfile = obj.gen_ldc_grid(rsrp1=rsrp1, rsrp2=rsrp2, sampling = 'kde',koi_table_folder=koi_table_folder,koi_table_filename=koi_table_filename)

    print('ldc_ratio_grid_file:',obj.ldc_ratio_grid_file)
    obj.execute()
    