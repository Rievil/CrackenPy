# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 08:47:44 2023

@author: dvorr
"""


import pandas as pd
import numpy as np
import math
from scipy.fft import fft
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import kurtosis
import re
import os

import sqlite3
import pickle
import json
import datetime
import time

from scipy.signal import get_window
from scipy.signal import filtfilt
from scipy.signal import butter

import scipy.io as sio
from scipy.signal import find_peaks


from multiprocessing import Pool, Process, Manager, freeze_support

import tqdm

def get_desc(desc):

    file=desc['name'].replace('.csv','')

    parts=file.split('-')
    desc['ID']=int(parts[0])
    desc['binder']=parts[1].lower()
    desc['b_type']=parts[2].lower()
    desc['age']=int(parts[3].replace('d',''))
    desc['f_type']=parts[-1].lower()
    
    return desc

def list_dir(folder):

    ext='csv'
    
    im_path=folder
    df=pd.DataFrame()
    for root, dirs, files in os.walk(im_path,topdown=False):
       for name in files:
           file_path   = root + '\\' + name
           created     = os.path.getctime(file_path)
           modified    = os.path.getmtime(file_path)
           
    
           date=datetime.datetime.fromtimestamp(modified)
           
           rows_desc={"folder":root,"name":name,
                             "modified":modified,"date":date,
                             'filename':file_path}
           
           rows_desc=get_desc(rows_desc)
           
           dfi=pd.DataFrame(rows_desc,index=[0])
           df=pd.concat([df,dfi],axis=0)
    
    df['order']=df['modified']-df['modified'].min()
    df=df.sort_values(by='order', ascending=True)
    df=df.reset_index()
    
    extension = df['name'].str.split('.', expand=True)
    idx=extension[1]==ext
    
    dff=df[idx]
    file_list=df
    return file_list



def ReadSignalIE(file):
    try:
        ds=pd.read_csv(file,header=3,delimiter=';',decimal=',')
        ds.columns=('Time','Signal')
    
        idx=(ds['Time'].values>-0.005) & (ds['Time'].values<0.1)
        ds=ds[idx]
        ds=ds.reset_index()
        
        samples=ds.shape[0]
        x=ds['Time'].values
        duration=x[-1]-x[0]
        period=duration/samples
        fs=1/period
        
        sig={'time':ds['Time'].values,'signal':ds['Signal'].values,
                          'period':period,'dur':duration,'fs':fs,'filename':file,'sucess':True}
        
        signal={'filename':file,
             'signal':sig,
             'sucess':True}
    except:
        signal={'filename':file,
             'signal':[],
             'sucess':False}
    # print('done')
    return signal

if __name__ == '__main__':
    freeze_support()
    pool = Pool(12)
    results = []

    file_list=list_dir(r'C:\Users\dvorr\VUT\22-02098S - General\Data\Malty-rezonance-signaly')
    df=file_list
    # print('Files read')
    
    start_time = time.time()
    df_collection = pool.map(ReadSignalIE, df['filename'])
    pool.close()
    pool.join()
    duration=time.time()-start_time
    print("Elapsed time: {:0.2f} s".format(duration))
    dfa=pd.DataFrame(df_collection)
    dfm=file_list.merge(dfa,left_on='filename',right_on='filename')
    dfm.to_pickle("IESignals.pkl")  