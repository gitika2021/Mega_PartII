import numpy as np
import matplotlib.pyplot as plt
import subprocess
import shape_utils,gen_ldc_ratio_grid,genlc_with_grid
import add_noise_to_lcs_files,processing_transit_region
import preproclc_hscaled,dataset_split
from pathlib import Path
from paths import *
from models import *
import os
from binary_classifier import *
from rich import print as prcolor
import re
from utils import extract_key
import ast
import shutil
import pandas as pd
###################################
class MLPreProcessing():
    def __init__(self,Num=1000,N=1,maps_path=None, rsrp1=5, rsrp2=10,nproc=4,train_frac=0.8,seed=None,maps_folder_str="10", test=None,fresh_run=False):
        self.Num = Num
        self.N = N
        self.nproc = nproc
        self.train_frac = train_frac
        self.seed = seed
        self.fresh_run = fresh_run
        
        # base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
        base_dir = Path(Base_Dir) / "Data" # this is actually data directory
        print("base_dir",base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir = base_dir
        self.maps_path = maps_path
        self.maps_folder_str = maps_folder_str
        self.test = test
      
        # shapes directory
        self.shape_dir = self.base_dir / f"OM{self.maps_folder_str}"
        self.shape_dir.mkdir(parents=True, exist_ok=True)
        self.shape_file = self.shape_dir / f"{self.N}.npy"
        
        self.koi_table_folder = Kepler_Dir
        self.koi_table_filename = KOI_Table_Filename
        self.kepler_error_file = self.koi_table_folder + Kepler_Error_Filename

        self.rsrp1 = rsrp1
        self.rsrp2 = rsrp2
        
        self.out_dir_lc = self.base_dir / f"LC{self.maps_folder_str}"
        self.out_dir_lc.mkdir(parents=True, exist_ok=True)
        #print('self.out_dir_lc',self.out_dir_lc)
        self.out_dir_proc_lc = self.out_dir_lc / "proc/RsRp_{0:d}_{1:d}".format(self.rsrp1,self.rsrp2)
        self.out_dir_proc_lc.mkdir(parents=True, exist_ok=True)
        self.out_stem_lc = self.out_dir_proc_lc / f"{self.N}"
        self.out_file_lc = str(self.out_stem_lc) + 'LC.npy' # this is now a string
        self.out_file_lc_meta = str(self.out_stem_lc) + '_meta.npy'
    
        self.out_dir_orig_lc = self.out_dir_lc / "orig/RsRp_{0:d}_{1:d}".format(self.rsrp1,self.rsrp2)
        self.out_dir_orig_lc.mkdir(parents=True, exist_ok=True)

        self.noisy_ltcrv_folder = self.out_dir_proc_lc / "Binned_LC"
        self.lc_hscaled_filename = f"{self.N}LC_hscaled"
        self.lc_hscaled_file = self.out_dir_proc_lc / (self.lc_hscaled_filename + ".npy")
        #print('self.lc_hscaled_file',self.lc_hscaled_file)
        self.train_dir = self.base_dir / "Train/RsRp_{0:d}_{1:d}".format(self.rsrp1,self.rsrp2)
        self.train_dir.mkdir(parents=True, exist_ok=True)

        self.model_dir = Path(Base_Dir + "Model/RsRp_{0:d}_{1:d}".format(self.rsrp1,self.rsrp2))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.figure_dir = Path(Base_Dir + "Figures/RsRp_{0:d}_{1:d}".format(self.rsrp1,self.rsrp2))
        self.figure_dir.mkdir(parents=True, exist_ok=True)

        # LDC and radius ratio directory
        self.ldc_dir = self.base_dir / "LDC_RSRP_GRIDS"
        self.ldc_dir.mkdir(parents=True, exist_ok=True)

        self.ldc_dir_bin = self.ldc_dir / "RsRp_{0:d}_{1:d}".format(self.rsrp1,self.rsrp2)
        self.ldc_dir_bin.mkdir(parents=True, exist_ok=True)
        self.ldc_ratio_grid_file = self.ldc_dir_bin / f"ldc_rsrp.npy"
        #self.ldc_ratio_grid_file = self.ldc_dir_bin / f"ldc_rsrp_{self.rsrp1}_{self.rsrp2}.npy"
  
        # self.ldc_ratio_grid_file = gen_ldc_ratio_grid.main(rsrp1=self.rsrp1,rsrp2=self.rsrp2,
        #                                                    koi_table_folder=self.koi_table_folder,
        #                                                    koi_table_filename=self.koi_table_filename,
        #                                                    base_dir=self.base_dir,fig_dir=self.figure_dir)
        
        print("... initialization complete.")

    
    def gen_ltcrv_ldc_grid_file(self):
        gen_ldc_ratio_grid.main(rsrp1=self.rsrp1,rsrp2=self.rsrp2,
                                koi_table_folder=self.koi_table_folder,
                                koi_table_filename=self.koi_table_filename,
                                outfile=self.ldc_ratio_grid_file,
                                fig_dir=self.figure_dir,random_seed=self.seed)
        return
                
    def gen_shapes(self):
        shape_utils.main(Num=self.Num,N=self.N,shape_dir=self.shape_dir,maps_path=self.maps_path) 
        return
                
    def gen_ltcrvs(self):        
        genlc_with_grid.main(self.N,1,self.base_dir,self.shape_dir,self.ldc_ratio_grid_file,
                             self.out_stem_lc,self.out_dir_orig_lc,
                             nproc=self.nproc,random_seed=self.seed )
        return

    def add_noise(self,seed =None):
        add_noise_to_lcs_files.main(self.out_file_lc,self.kepler_error_file,self.figure_dir, 
                                   random_seed=seed)
        return

    def select_transit_region(self):
        trs = processing_transit_region.TransitRegionSelector(ltcrv_files_folder=self.noisy_ltcrv_folder,
                                                              max_workers=self.nproc)
        trs.find_transit_region_and_save_parallel(N=self.N)
        processing_transit_region.combine_flux(self.noisy_ltcrv_folder, 
                                               self.N, output_file=self.lc_hscaled_filename+".npy",
                                               savefolder_path=self.out_dir_proc_lc)        
        return

    def preprocess_ltcrvs(self):
        hscaled_processed_file = preproclc_hscaled.main(lc_hscaled_path=str(self.out_dir_proc_lc)+"/"+self.lc_hscaled_filename)
        return hscaled_processed_file
    
    def split_dataset(self,hscaled_processed_file):
        dataset_split.main(N=self.N,lc_path=hscaled_processed_file,img_path=str(self.shape_file),
                           train_dir=self.train_dir,train_frac=self.train_frac,seed=self.seed)
        return 


    def execute(self):
        prcolor("[bold green]If testing pipeline set: --fresh_run = 1")
        if self.fresh_run==True:
            print("Running pipeline afresh")
        
            for file in Path(self.noisy_ltcrv_folder).glob(f"{self.N}LC*.npz"):
                file.unlink()
                #print(f"Deleted {file}")
    
            self.gen_ltcrv_ldc_grid_file()
            self.gen_shapes()
            prcolor("[bold green]Generated Bezier shapes")
            if self.test == True:
                SHAPE_SIZE = shape_utils.SHAPE_SIZE
                shape_circle = shape_utils.generate_circles(num_maps=1, size=SHAPE_SIZE)
                manual_shapes = np.load("weird_test_shapes_solid.npy")
                test_shapes_all = np.concatenate((np.load(self.shape_file), 
                                                      shape_circle,manual_shapes))
                self.Num = self.Num+len(shape_circle)+len(manual_shapes)
                np.save(self.shape_file,test_shapes_all)
                prcolor("[bold green]Added manual generated shapes in case of test data")
             
            self.gen_ltcrvs()
            prcolor("[bold green]Generated light curves")
            self.add_noise(seed=self.seed)
            prcolor("[bold green]Added noise")
            self.select_transit_region()
            prcolor("[bold green]Selected transit region")
            hscaled_processed_file = self.preprocess_ltcrvs()
            prcolor("[bold green]Final processed light curves generated")
            #self.split_dataset(hscaled_processed_file)
            
        elif self.fresh_run==False:
            print("Running pipeline")
            prcolor("[bold red] fresh_run = False: Num arg on command line ignored in case shape and light curve files pre-exist. ")
            # files = sorted(
            #             [
            #                 f for f in Path(self.noisy_ltcrv_folder).glob(f"{self.N}LC*.npz")
            #                 if (m := re.search(rf"{self.N}LC(\d+)", f.stem)) and int(m.group(1)) >= self.Num
            #             ],
            #             key=lambda f: int(re.search(rf"{self.N}LC(\d+)", f.stem).group(1))
            #         )
            
            # for file in files:
            #     file.unlink()
                
            if not Path(self.ldc_ratio_grid_file).is_file():
                print("Generating ldc ratio grid")
                self.gen_ltcrv_ldc_grid_file()        
            
            if not self.shape_file.is_file():                
                print("Generating Shapes")
                self.gen_shapes()  
                if self.test == True:
                    SHAPE_SIZE = shape_utils.SHAPE_SIZE
                    shape_circle = shape_utils.generate_circles(num_maps=1, size=SHAPE_SIZE)
                    manual_shapes = np.load("weird_test_shapes_solid.npy")
                    test_shapes_all = np.concatenate((np.load(self.shape_file), 
                                                      shape_circle,manual_shapes))
                    self.Num = self.Num+len(shape_circle)+len(manual_shapes)
                    np.save(self.shape_file,test_shapes_all)
                    prcolor("[bold green]Added manual generated shapes in case of test data")
                    
                files = sorted(
                        [
                            f for f in Path(self.noisy_ltcrv_folder).glob(f"{self.N}LC*.npz")
                            if (m := re.search(rf"{self.N}LC(\d+)", f.stem)) and int(m.group(1)) >= self.Num
                        ],
                        key=lambda f: int(re.search(rf"{self.N}LC(\d+)", f.stem).group(1))
                    )
                            
                for file in files:
                    file.unlink()                
                #since shapes are updated light curves have to be regenarated
                self.gen_ltcrvs()
                self.add_noise(seed=self.seed)
                self.select_transit_region()
                hscaled_processed_file = self.preprocess_ltcrvs()

            else:                                    
                if not (self.out_dir_proc_lc / f"{self.N}LC.npy").is_file():
                    prcolor("[bold green]GShapes files exist, generating light curves")
                    self.gen_ltcrvs()
                    
                #self.add_noise()
                if Path(self.out_file_lc).is_file():
                    prcolor("[bold green]Adding noise to light curves")
                    self.add_noise(seed=self.seed)
        
                if not self.lc_hscaled_file.is_file():
                    prcolor("[bold green]Select Transit region")
                    self.select_transit_region()
        
                if self.lc_hscaled_file.is_file():
                    prcolor("[bold green]Preprocess light curves")
                    hscaled_processed_file = self.preprocess_ltcrvs()
                else:
                    hscaled_processed_file = None
        
        if hscaled_processed_file is not None and self.test==False:
            if Path(hscaled_processed_file).is_file():
                prcolor("[bold green]Split dataset into train and val")
                self.split_dataset(hscaled_processed_file)
                
        return
###################################


###################################
class MLInference():
    def __init__(self,lc_dir=None, maps_dir=None, nproc=4, rsrp1=5, rsrp2=10,n_scale=2, N=None,nrpoc=4, obj=None ):
        self.lc_dir = lc_dir
        self.lc_dir_pthobj = Path(self.lc_dir) 
        self.N = N
        self.mlprep = obj
        
        self.nproc = nproc
        self.rsrp1 = rsrp1
        self.rsrp2 = rsrp2
        self.n_scale = n_scale
        
        #self.mlprep = MLPreProcessing(rsrp1=self.rsrp1, rsrp2=self.rsrp2, N=self.N)
        self.model = self.mlprep.model_dir / f"model_n{self.n_scale}.pth"
        print(f"Model used for inference: {self.model}")

        self.figure_dir = self.mlprep.figure_dir
        # self.figure_dir = Path(Base_Dir + "Figures/RsRp_{0:d}_{1:d}".format(self.rsrp1,self.rsrp2))
        # self.figure_dir.mkdir(parents=True, exist_ok=True)

        self.lcs_filename = self.lc_dir_pthobj / "light_curves_all.npy"
        self.key_filename = self.lc_dir_pthobj / "keynames_all.npy"
        self.pre_filename = self.lc_dir_pthobj / "prediction_maps_all.npy"
        self.map_filename = self.lc_dir_pthobj / "original_maps_all.npy"

        
    def extract_key(self,filepath,split_str="_binned.npz"):
        filename = os.path.basename(filepath)
        #filename = filepath.stem
        #print('filename',filename)
        return filename.split(split_str)[0]
    
    def process_orig_ltcrvs(self):
        if self.N is None:
            trs=processing_transit_region.TransitRegionSelector(ltcrv_files_folder=self.lc_dir,
                                                                max_workers=self.nproc)
            trs.find_transit_region_and_save_parallel()
            ltcrv_npz_files = list(self.lc_dir_pthobj.glob(f"*_binned_transit_interp.npz"))

            self.orig_shapes = np.zeros((len(ltcrv_npz_files),shape_utils.SHAPE_SIZE,shape_utils.SHAPE_SIZE))
            np.save(self.map_filename,self.orig_shapes)
            prcolor('[bold green]original shapes not applicable: fake data created')
        else:
                    
            # for file in Path(self.lc_dir_pthobj).glob(f"{self.N}*_binned_transit_interp.npz"):
            #     file.unlink()
            ltcrv_npz_files = list(self.lc_dir_pthobj.glob(f"{self.N}*_binned_transit_interp.npz"))
            print(f'Number of Test shapes is {len(ltcrv_npz_files)}')
            orig_shapes_file = self.mlprep.shape_file
            orig_shapes = np.load(orig_shapes_file)
            np.save(self.map_filename,orig_shapes)
            print(f'Saved Original Test shapes to {self.map_filename}')
            prcolor('[bold green]original shapes applicable: actual data used')

        files_sorted = sorted(ltcrv_npz_files, key=processing_transit_region.extract_index)
        print('files_sorted',files_sorted)
        data_temp = np.load(ltcrv_npz_files[0])
        
        keys_sorted = np.empty(len(ltcrv_npz_files), dtype=object)
        lc_sorted = np.zeros((len(ltcrv_npz_files),data_temp['flux'].shape[0]))
        
        for i, f in enumerate(files_sorted):
            key = self.extract_key(f,split_str="_binned_transit_interp.npz") 
            keys_sorted[i] = key
            data = np.load(f)
            lc_sorted[i,:] =data['flux']

        np.save(self.lcs_filename,lc_sorted)
        np.save(self.key_filename,keys_sorted)     
        return
        
    def read_processed_ltcrvs(self,pattern="kplr*"):
        trs=processing_transit_region.TransitRegionSelector()
        trs.load_and_plot_matched_ltcrvs(self.lc_dir,self.lc_dir,self.lc_dir,self.lc_dir,
            pattern=pattern,x_key="time",y_key="flux",show_plot=False,
            save_dir=self.figure_dir, N_plots = None)
        return
                
    def infer_2dshape(self):   
        generator = HybridConvNet(n=self.n_scale)
        generator.load_state_dict(torch.load(self.model,weights_only=True, 
                                             map_location='cpu'))
   
        lc=torch.tensor(np.load(self.lcs_filename))
        out=[]
        #print(lc.shape)
        generator.eval()
        for i in range(lc.shape[0]):
            #out.append(generator(lc[i].squeeze() ).squeeze().detach().cpu())
            out.append(generator(lc[i].unsqueeze(0).float()).squeeze().detach().cpu())
        np.save(self.pre_filename,torch.stack(out, dim=0).numpy())        
        return
 

    def plot_prediction_orig_maps(self):
        images = self.orig_shapes
        predictions = self.pred_shapes
        inpt_ltcrvs = self.inpt_ltcrvs
        
        n_total = images.shape[0]
        n_cols = 12                               # number of images per row
        n_groups = int(np.ceil(n_total / n_cols))  # number of full groups
        
        for group in range(n_groups):
            start = group * n_cols
            end = min(start + n_cols, n_total)
            count = end - start
        
            fig, axes = plt.subplots(3, count, figsize=(count * 1.2, 2.5), constrained_layout=True)
        
            for i in range(count):
                # Plot image
                ax_img = axes[0, i] if count > 1 else axes[0]
                ax_img.imshow(images[start + i], cmap='viridis')
                #ax_img.set_title(f"{kepnames[start + i]}\n$R_p/R_s$: {rp_rs_ratio[start + i]:.2f}\n$snr$: {snr[start + i]:.2f}", fontsize=8)
                ax_img.axis('off')

                ax_pred = axes[1, i] if count > 1 else axes[0]
                ax_pred.imshow(predictions[start + i], cmap='viridis')
                #ax_img.set_title(f"{kepnames[start + i]}\n$R_p/R_s$: {rp_rs_ratio[start + i]:.2f}\n$snr$: {snr[start + i]:.2f}", fontsize=8)
                ax_img.set_title(f"{self.keynames[start + i]}\n", fontsize=6)
                ax_pred.axis('off')
                
                # Plot 1D profile
                ax_prof = axes[2, i] if count > 1 else axes[1]
                ax_prof.plot(inpt_ltcrvs[start + i])
                ax_prof.set_xticks([])
                ax_prof.set_yticks([])
            if group+1 > 20 and self.N is not None:
                plt.suptitle(f"Images and 1D Profiles: Group {group+1}/{n_groups}", fontsize=12)
                plt.tight_layout()
                plt.savefig(self.figure_dir/ f'{self.N}_grp{group+1}_org_vs_pred.png')
                plt.show()
            elif self.N is None:
                
                plt.suptitle(f"Images and 1D Profiles: Group {group+1}/{n_groups}", fontsize=12)
                plt.tight_layout()
                plt.savefig(self.figure_dir/ f'{self.N}_grp{group+1}_org_vs_pred.png')
                plt.show()
                
                
        return
        

    def load_imp_files(self):
        self.orig_shapes = np.load(self.map_filename)
        self.pred_shapes = np.load(self.pre_filename)
        self.inpt_ltcrvs = np.load(self.lcs_filename)
        self.keynames = np.load(self.key_filename,allow_pickle=True)
        
        prcolor("[bold green]Check that sizes of these files are same??")
        print(f"Shape of original file:{self.orig_shapes.shape}")
        print(f"Shape of predicted file:{self.pred_shapes.shape}")
        print(f"Shape of light curve file:{self.inpt_ltcrvs.shape}")
        print(f"Shape of keynames file:{self.keynames.shape}")
        print('self.keynames',self.keynames)
        return
 
    def run_binary_classifier(self):        
        self.flux_cutI = 0.94
        self.dist_cutII = 0.10
        self.deviation_estimator = 'flux'
        
        if self.N is None:
            # for real data
            self.binclas_pred,acc_pred,self.CI_pred,self.CII_pred,flags_str_pred,self.rad_fit_pred, self.fmaps_pred = batch_predict_shape(images=self.pred_shapes,deviation_estimator=self.deviation_estimator,cut1=self.flux_cutI, cut2=self.dist_cutII,num_cpus=self.nproc, show= False)  
            
            self.CI_pred = np.round(self.CI_pred,2)
            self.CII_pred = np.round(self.CII_pred,2)
        
        else:
            # for simulated test data              
            binclas_orig,acc,CI,CII,flags_str,self.rad_fit, self.fmaps = batch_predict_shape(images=self.orig_shapes,deviation_estimator=self.deviation_estimator,cut1=self.flux_cutI, cut2=self.dist_cutII,num_cpus=self.nproc, show= False) 
    
            self.binclas_pred,acc_pred,self.CI_pred,self.CII_pred,flags_str_pred,self.rad_fit_pred, self.fmaps_pred = batch_predict_shape(images=self.pred_shapes,deviation_estimator=self.deviation_estimator,cut1=self.flux_cutI, cut2=self.dist_cutII,num_cpus=self.nproc, show= False)
            
            savefigname = self.figure_dir/ f'confusion_met_{self.N}.png'
            results = estimate_ml_metrics(binclas_orig, binclas_pred, 
                            savefig=savefigname)
            


        return

    def create_anomaly_catalog(self):
        planet_names = np.array(['K' + key[15:20] + '-' + key[21:] for key in self.keynames])
        planet_kic = np.array(['KIC' + key[4:12] for key in self.keynames])
        
        anomalous_objs_mask = (self.binclas_pred == 1)
        
        anomalous_objs_keys = self.keynames[anomalous_objs_mask]
        anomalous_planet_names = planet_names[anomalous_objs_mask]
        anomalous_planet_kic = planet_kic[anomalous_objs_mask]
        #anomalous_planet_names = ['K' + key[15:20] + '-' + key[21:-1]+' '+key[-1] for key in self.keynames]
        anomalous_objs_CI = self.CI_pred[anomalous_objs_mask]
        anomalous_objs_CII = self.CII_pred[anomalous_objs_mask]
        anomalous_objs_maps = self.fmaps_pred[anomalous_objs_mask, :, :]
        print('anomalous_objs_keys',anomalous_objs_keys,anomalous_planet_names) 
        print('anomalous_planet_kic',anomalous_planet_kic)

        planet_objs_keys = self.keynames[~anomalous_objs_mask]
        planet_planet_names = planet_names[~anomalous_objs_mask]
        planet_planet_kic = planet_kic[~anomalous_objs_mask]
        #planet_planet_names = ['K' + key[15:20] + '-' + key[21:-1]+' '+key[-1] for key in self.keynames]
        planet_objs_CI = self.CI_pred[~anomalous_objs_mask]
        planet_objs_CII = self.CII_pred[~anomalous_objs_mask]
        planet_objs_maps = self.fmaps_pred[~anomalous_objs_mask, :, :]
        print('planet_objs_keys',planet_objs_keys,planet_planet_names) 
        print('planet_planet_kic',planet_planet_kic)
        
        np.save(self.lc_dir_pthobj / f"{self.N}_final_predicted_maps_anamalous.npy",anomalous_objs_maps)
        np.save(self.lc_dir_pthobj / f"{self.N}_final_predicted_maps_planet.npy",planet_objs_maps)
        
        # Create DataFrame
        df = pd.DataFrame({
            "KIC ID": anomalous_planet_kic,
            "Planet Names": anomalous_planet_names,
            "Cut I": anomalous_objs_CI,
            "Cut II": anomalous_objs_CII
        })
        
        print(df)
        
        # ---- Save table as PNG ----
        # fig, ax = plt.subplots(figsize=(5, 2))
        # ax.axis('off')
        
        # table = ax.table(
        #     cellText=df.values,
        #     colLabels=df.columns,
        #     loc='center',
        #     cellLoc='center'
        # )
        
        # table.auto_set_font_size(False)
        # table.set_fontsize(10)
        # table.scale(1.2, 1.5)
        
        # # Make header bold
        # for (row, col), cell in table.get_celld().items():
        #     if row == 0:
        #         cell.set_text_props(weight='bold')
        
        # plt.savefig(self.figure_dir/ f'{self.N}_kepler_anomalous_table.png', bbox_inches='tight', dpi=300)
        # plt.close()

        fig, ax = plt.subplots(figsize=(7, 2.5), dpi=300)
        ax.axis('off')
        
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            loc='center',
            cellLoc='center'
        )
        
        # Font control
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        
        # Scale table
        table.scale(1.2, 1.6)
        
        # Style cells
        for (row, col), cell in table.get_celld().items():
        
            # Thin borders
            cell.set_linewidth(0.6)
        
            # Header row
            if row == 0:
                cell.set_text_props(weight='bold', color='black')
                cell.set_facecolor('#D9E6F2')   # light blue-gray
                cell.set_height(0.18)
        
            else:
                cell.set_facecolor('white')
                cell.set_height(0.15)
        
        # Optional title
        plt.title(
            'Kepler Anmalous Candidates',
            fontsize=13,
            weight='bold',
            pad=12
        )
        
        plt.tight_layout()
        
        # Save for paper
        plt.savefig(
            self.figure_dir/ f'{self.N}_kepler_anomalous_table.png',
            bbox_inches='tight',
            dpi=300
        )
        
        plt.show()

        return        
        
    def execute(self):
        if self.lc_dir is not None:
            self.process_orig_ltcrvs()
            if self.N is not None:
                pattern = f"{self.N}*"
            else:
                pattern =f"kplr*"
            self.read_processed_ltcrvs(pattern=pattern)
            self.infer_2dshape()
            self.load_imp_files()
            self.plot_prediction_orig_maps()
            self.run_binary_classifier()
            if self.N is None:
                self.create_anomaly_catalog()

        return
###################################

###################################

class LogBinHistogram():
    """
    Create and overlay logarithmic histograms of Rp/Rs.
    """

    def __init__(self, nbins=15, figsize=(7, 5), figure_dir = None):

        self.nbins = nbins
        self.figure_dir = figure_dir

        # # Figure for Rp/Rs
        # self.fig_ratio, self.ax_ratio = plt.subplots(figsize=figsize)

        # self.ax_ratio.set_xscale('log')
        # self.ax_ratio.set_xlabel("Rp/Rs")
        # self.ax_ratio.set_ylabel("Counts")
        # self.ax_ratio.grid(True, which='both', alpha=0.3)

        # Figure for inverse ratio
        self.fig_inv, self.ax_inv = plt.subplots(figsize=figsize)

        self.ax_inv.set_xscale('log')
        self.ax_inv.set_xlabel("1 / (Rp/Rs)")
        self.ax_inv.set_ylabel("Counts")
        self.ax_inv.grid(True, which='both', alpha=0.3)

    def add(
        self,
        ratio,
        label=None,
        alpha=0.5,
        color=None,
        outfile=None,
        show_points=True,
        show_bar=True,
        snr = None
    ):
        """
        Add histogram to existing plots.

        Parameters
        ----------
        ratio : array-like
            Rp/Rs values.
        label : str
            Legend label.
        alpha : float
            Transparency.
        color : str
            Plot color.
        outfile : str
            Save inverse bins if provided.
        """

        ratio = np.asarray(ratio)

        rmin = np.min(ratio)
        rmax = np.max(ratio)

        # Logarithmic bins
        bin_edges = np.logspace(
            np.log10(rmin),
            np.log10(rmax),
            self.nbins + 1
        )
        #print('bin_edges',bin_edges)
        # Counts
        counts, _ = np.histogram(ratio, bins=bin_edges)

        # ---SNR sorting--------
        # Bin index for each ratio value
        # values are in [0, nbins-1]
        bin_idx = np.digitize(ratio, bin_edges) - 1
        
        # Handle edge case where ratio == rmax
        bin_idx[bin_idx == self.nbins] = self.nbins - 1
        
        # Store [min_snr, max_snr] for each bin
        snr_ranges = np.zeros((self.nbins, 3))
        
        for i in range(self.nbins):
        
            # SNR values belonging to this ratio bin
            snr_in_bin = snr[bin_idx == i]
        
            if len(snr_in_bin) > 0 and counts[i]!=0:
        
                snr_ranges[i, 0] = np.min(snr_in_bin)
                snr_ranges[i, 1] = np.max(snr_in_bin)
                snr_ranges[i, 2] = int(np.percentile(snr_in_bin,50))
        
            else:
                snr_ranges[i] = [-np.inf, np.inf,np.inf]
        
        # ----- Rp/Rs plots -----

        # Geometric centers
        bin_centers = np.sqrt(
            bin_edges[:-1] * bin_edges[1:]
        )

        # Linear widths
        bin_widths = np.diff(bin_edges)

        # # Line plot
        # if show_points:

        #     self.ax_ratio.plot(
        #         bin_centers,
        #         counts,
        #         marker='o',
        #         label=label,
        #         color=color
        #     )

        # # Bar plot
        # if show_bar:

        #     self.ax_ratio.bar(
        #         bin_centers,
        #         counts,
        #         width=bin_widths,
        #         align='center',
        #         edgecolor='black',
        #         alpha=alpha,
        #         color=color
        #     )

        # ----- Inverse ratio bins -----

        inv_edges = 1.0 / bin_edges
        #print('inv_edges',inv_edges)
        inverse_bins = np.zeros(
            (self.nbins, 2),
            dtype=int
        )

        for i in range(self.nbins):

            left = min(
                inv_edges[i],
                inv_edges[i + 1]
            )

            right = max(
                inv_edges[i],
                inv_edges[i + 1]
            )

            inverse_bins[i, 0] = int(round(left))
            inverse_bins[i, 1] = int(round(right))

        # Sort increasing
        order = np.argsort(inverse_bins[:, 0])

        inverse_bins = inverse_bins[order]
        counts = counts[order]
        snr_ranges = snr_ranges[order]
        # Save if requested
        if outfile is not None:

            np.savetxt(
                outfile,
                inverse_bins,
                fmt="%d",
                header="left_edge(1/ratio) right_edge(1/ratio)"
            )

        # Centers and widths
        inv_centers = 0.5 * (
            inverse_bins[:, 0]
            + inverse_bins[:, 1]
        )

        inv_widths = (
            inverse_bins[:, 1]
            - inverse_bins[:, 0]
        )

        # Plot inverse histogram
        if show_points:

            self.ax_inv.plot(
                inv_centers,
                counts,
                marker='o',
                label=label,
                color=color
            )
        self.ax_inv.bar(
            inv_centers,
            counts,
            width=inv_widths,
            align='center',
            edgecolor='black',
            alpha=alpha,
            color=color,
            label=''
        )
        #print('inverse_bins',inverse_bins[0])
        return inverse_bins, counts,snr_ranges

    def show(self):

        #self.ax_ratio.legend()
        self.ax_inv.legend()

        #self.fig_ratio.tight_layout()
        self.fig_inv.tight_layout()
        if self.figure_dir is not None:
            plt.savefig(
            self.figure_dir / "kepler_rsrp_distribution.png",
            dpi=500,
            bbox_inches='tight',
            pad_inches=0.2
            )
        plt.show()

        return
###################################

###################################
# class for processing Kepler Data

class SaveKeplerToRsRpBins():    
    def __init__(self,lcs_dir,snr_cut = 50, n_rsrp_bins=21):
        import kepler_distribution_utils
        self.filesfolder = Path(lcs_dir)
        self.snr_cut = snr_cut
        self.n_rsrp_bins = n_rsrp_bins
        self.obj = MLPreProcessing()
        table_subset_new = kepler_distribution_utils.main(snr_cut = self.snr_cut, n_rsrp_bins=self.n_rsrp_bins)
        self.df = table_subset_new
                
    def distribute_kepler_into_rsrp_bins(self):
        ltcrv_npz_files = list(self.filesfolder.glob("*_binned.npz"))
        objs_not_found = []
        objs_found = []
        for i, f in enumerate(ltcrv_npz_files):
            #print('file',f)
            key = extract_key(f,split_str="_binned.npz")
            planet_name = key[14:]
            planet_name = 'K'+planet_name[1:6]+'-'+planet_name[7:]
            #planet_name= 'Kepler_809b'

            matches = self.df.loc[self.df['kepname'] == planet_name, 'invrprs_bin_edges']
            if not matches.empty:
                objs_found.append(planet_name)
                rsrp_bin = matches.iloc[0]
                #print('rsrp_bin',rsrp_bin)
                rsrp_bin = ast.literal_eval(rsrp_bin)
                rsrp1, rsrp2 = rsrp_bin
                #print('rsrp1, rsrp2',int(rsrp1), int(rsrp2),type(rsrp1))

                # copy phase folded files to Binned directory as per their rsrp ratio
                kepler_binned_dir = Path(Base_Dir) / f"Kepler_RsRp_Bins/RsRp_{rsrp1}_{rsrp2}"
                kepler_binned_dir.mkdir(parents=True, exist_ok=True)
                src = Path(f)
                #shutil.copy2(src, kepler_binned_dir / src.name)
                try:
                    shutil.copy2(src, kepler_binned_dir / src.name)
                except PermissionError:
                    shutil.copy(src, kepler_binned_dir / src.name)
                                                
            else:
                #print("No matching kepname found")
                objs_not_found.append(planet_name)
        print('Kepler objs_not_found and not Binned in RsRp',len(objs_not_found))
        print('Kepler objs_found and binned in RsRp',len(objs_found))
            
        return 
###################################