
# Python script for KOI Table related things
# Import required libraries
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
from numpy import array,append,arange,zeros,exp,sin,random,std
import pandas as pd



class KoiTableObjs():

  def __init__(self,files_dir = "/content/drive/MyDrive/Megastructure-2025/CSV-Files/",koi_file_name = "koi_cumulative_2025.06.28_01.24.15.csv",verbose=False):

    self.files_dir = files_dir    # "/content/drive/MyDrive/Megastructure-2025/CSV-Files/"
    self.koi_file_name = koi_file_name    # "koi_cumulative_2025.06.28_01.24.15.csv"
    self.koi_table = pd.read_csv(self.files_dir+self.koi_file_name,comment='#')
    self.verbose = verbose

    # add a column ['kepid_str'] to koi table containing kplr000000000
    kepid = self.koi_table['kepid'].to_numpy()
    kepname = self.koi_table['kepler_name'].to_numpy()

    kepname_modfied = np.char.replace(kepname.astype(str), ' ', '')
    self.verbose and print(f'Modified Kepler name column added to the table{kepname_modfied[0:5]}')


    # Format with zero-padding and prefix using vectorized operations
    # kepid_modified = np._core.defchararray.add('kplr',np.char.zfill(kepid.astype(str), 9))
    kepid_modified = np.char.add('kplr', np.char.zfill(kepid.astype(str), 9))
    self.verbose and print(f'Modified Kepler id column added to the table{kepid_modified[0:5]}')


    # Store this as a new column in koi table
    self.koi_table['kepid_str'] = kepid_modified
    self.koi_table['kepname'] = kepname_modfied
    #print("self.koi_table['kepid_str']",self.koi_table['kepid_str'])


  def get_koi_confirmed(self):
    koi_conf_plans = self.koi_table['kepler_name'].replace('', pd.NA).dropna()
    idx_temp = np.where(np.isin(self.koi_table['kepler_name'],koi_conf_plans))[0]
    koi_conf_plans_tab = self.koi_table.iloc[idx_temp]

    self.N_koi_with_kepnames = len(koi_conf_plans_tab)
    self.koi_conf_plans_tab = koi_conf_plans_tab


  def get_koi_candidates(self):
    self.candidate_kois = self.koi_table[self.koi_table['koi_disposition']=='CANDIDATE']
    self.n_candidate_kois = len(self.candidate_kois)

  def get_koi_fps(self):
    self.false_pos_kois = self.koi_table[self.koi_table['koi_disposition']=='FALSE POSITIVE']
    self.n_fp_kois = len(self.false_pos_kois)

  def choose_koi_cols(self, tabel='', use_cols = ['kepid','koi_count','kepler_name','koi_period','koi_model_snr','koi_ror']):
    table_arr = table[use_cols].to_numpy()
    print('table_arr',table_arr)

  def print_tabinfo():
    print('\033[1;39m Number of confirmed planets in KOI Table \033[0m',self.N_koi_with_kepnames)
    print('\033[1;39m Number of False positive in KOI Table \033[0m',self.n_fp_kois)
    print('\033[1;39m Number of candidates in KOI Table \033[0m',self.n_candidate_kois)
