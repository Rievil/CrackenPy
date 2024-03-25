# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 11:05:03 2023

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

from tqdm import tqdm


dfa = pd.read_pickle("Signals\IESignals.pkl")  
#%%

dfl=dfa[(dfa['f_type']=='l') & (dfa['sucess']==True)][['filename','ID','age','binder','date','signal']]
dft=dfa[(dfa['f_type']=='t') & (dfa['sucess']==True)][['ID','age','binder','signal']]
dff=dfa[(dfa['f_type']=='f') & (dfa['sucess']==True)][['ID','age','binder','signal']]

dfl=dfl.rename(columns={'signal':'signal_l'})
dft=dft.rename(columns={'signal':'signal_t'})
dff=dff.rename(columns={'signal':'signal_f'})

dfs=[dfl,dft,dff]


df=dfl.merge(dft,right_on=['ID','age','binder'],left_on=['ID','age','binder'])
df=df.merge(dff,right_on=['ID','age','binder'],left_on=['ID','age','binder'])

#%%





#%%

from IEFrame import ResMeas

dd=pd.DataFrame()

for row in tqdm(range(0,df.shape[0])):
    rem=ResMeas(fl=df['signal_l'][row],ft=df['signal_t'][row],ff=df['signal_f'][row])
    rem.GetSpectrum()
    
    try:
        dom=rem.GetDom()
        dom['success']=True
    except:
        dom={'ff':0,'ft':0,'fl':0,'success':False}
    
    
    dom['RMS_ff']=rem.ff['sig_param']['rms']
    dom['RMS_ft']=rem.ft['sig_param']['rms']
    dom['RMS_fl']=rem.fl['sig_param']['rms']
    
    dom['ID']=df['ID'][row]
    dom['age']=df['age'][row]
    dom['binder']=df['binder'][row]
    dom['date']=df['date'][row]

    ddn=pd.DataFrame(dom,index=[row])
    dd=pd.concat([dd,ddn],axis=0)
#%
print("\nSuccess rate {:0.2f}%".format(dd['success'].values.sum()/dd.shape[0]))
#%%

mask1=(dd['ID']==31) & (dd['binder']=='h') | (dd['ID']==32) & (dd['binder']=='h') | (dd['ID']==33) & (dd['binder']=='h')
mask2=(dd['ID']==31) & (dd['binder']=='c') | (dd['ID']==32) & (dd['binder']=='c') | (dd['ID']==33) & (dd['binder']=='c')
mask3=(dd['ID']==26) & (dd['binder']=='wg') | (dd['ID']==27) & (dd['binder']=='wg') | (dd['ID']==29) & (dd['binder']=='wg')

tmask=(mask1) | (mask2) | (mask3)
dn=dd[~tmask]

#%
import seaborn as sns

sns.scatterplot(dn,x='RMS_ft',hue='binder',y='ft')
#%%
row=851
rem=ResMeas(fl=df['signal_l'][row],ft=df['signal_t'][row],ff=df['signal_f'][row])
rem.GetSpectrum()

dm=rem.GetDom()
for sig in [rem.fl,rem.ft,rem.ff]:
    spec=sig['spectrum']
    x=spec['xf']
    y=spec['yf']
    plt.plot(x,y)
#%%

ds = pd.pivot_table(dn, values=['fl','ft','ff'],columns=['age'], aggfunc="mean")
#%%

from scipy.optimize import curve_fit

mask1=(dd['ID']==31) & (dd['binder']=='h') | (dd['ID']==32) & (dd['binder']=='h') | (dd['ID']==33) & (dd['binder']=='h')
mask2=(dd['ID']==31) & (dd['binder']=='c') | (dd['ID']==32) & (dd['binder']=='c') | (dd['ID']==33) & (dd['binder']=='c')
mask3=(dd['ID']==26) & (dd['binder']=='wg') | (dd['ID']==27) & (dd['binder']=='wg') | (dd['ID']==29) & (dd['binder']=='wg')


fig,(ax1,ax2,ax3)=plt.subplots(3,1)
for mask,ax in zip([mask1,mask2,mask3],[ax1,ax2,ax3]):
    ds=dd[mask]
    ds=ds.sort_values(by='age')
    ds = pd.pivot_table(ds, values=['fl','ft','ff'],columns=['age'], aggfunc="mean")
    ds=ds.T
    
    x=ds.index
    
    for ftype in ['fl','ff','ft']:
        yl=ds[:][ftype].values
        
        # popt, pcov = curve_fit(func, x, yl)
        
        ax.plot(x,yl,label=ftype)
    
    ax.set_ylim(1000,14000)
    # ax.set_xlim(0,20)
    ax.legend()

#%%

from scipy.optimize import curve_fit

mask1=(dd['ID']==31) & (dd['binder']=='h') | (dd['ID']==32) & (dd['binder']=='h') | (dd['ID']==33) & (dd['binder']=='h')
mask2=(dd['ID']==31) & (dd['binder']=='c') | (dd['ID']==32) & (dd['binder']=='c') | (dd['ID']==33) & (dd['binder']=='c')
mask3=(dd['ID']==26) & (dd['binder']=='wg') | (dd['ID']==27) & (dd['binder']=='wg') | (dd['ID']==29) & (dd['binder']=='wg')

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

fig,(ax1,ax2,ax3)=plt.subplots(3,1)
for mask,ax in zip([mask1,mask2,mask3],[ax1,ax2,ax3]):
    ds=dd[mask]
    ds=ds.sort_values(by='age')
    ds = pd.pivot_table(ds, values=['fl','ft','ff'],columns=['age'], aggfunc="mean")
    ds=ds.T
    
    x=ds.index
    

    yl=ds[:]['fl'].values
    yf=ds[:]['ff'].values
    yt=ds[:]['ft'].values
    
    rat1=yt/yf
    rat2=yl/yt
    rat3=yl/yf
    
    # popt, pcov = curve_fit(func, x, yl)
    
    ax.plot(x,rat1,label='t:f')
    ax.plot(x,rat2,label='l:t')
    ax.plot(x,rat3,label='l:f')
    
    # ax.set_ylim(1000,14000)
    ax.legend()


    