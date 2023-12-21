# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 16:54:37 2023

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

class IESolver:
    def __init__(self):
        
        pass
    
    def get_desc(self,desc):

        file=desc['name'].replace('.csv','')

        parts=file.split('-')
        desc['ID']=int(parts[0])
        desc['binder']=parts[1].lower()
        desc['b_type']=parts[2].lower()
        desc['age']=int(parts[3].replace('d',''))
        desc['f_type']=parts[-1].lower()
        
        return desc
    
    def list_dir(self,folder):
        self.folder=folder
        self.extension='csv'
        
        im_path=self.folder
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
               
               rows_desc=self.get_desc(rows_desc)
               
               dfi=pd.DataFrame(rows_desc,index=[0])
               df=pd.concat([df,dfi],axis=0)
        
        df['order']=df['modified']-df['modified'].min()
        df=df.sort_values(by='order', ascending=True)
        df=df.reset_index()
        
        extension = df['name'].str.split('.', expand=True)
        idx=extension[1]==self.extension
        
        dff=df[idx]
        self.file_list=df
        return self.file_list
    
    def read_multi(self):
        freeze_support()
        pool = Pool(12)
        results = []
        df=self.file_list
        
        start_time = time.time()
        df_collection = pool.map(ReadSignalIE, df['filename'])
        pool.close()
        pool.join()
        print("Elapsed time: {:0.2f} s".format(time.time()-start_time))
        dfa=pd.DataFrame(df_collection)
        dfm=self.file_list.merge(dfa,left_on='filename',right_on='filename')
        dfm.to_pickle("IESignals.pkl")  
        self.signal_list=dfm
        # print(results)
    
    
    def ReadSignal(self,row,filename=None):
        if filename is not None:
            file=filename
        else:
            file=r"{:s}\{:s}".format(self.file_list['folder'][row],self.file_list['name'][row])
        
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
        
        sig={'time':ds['Time'].values,'signal':ds['Signal'].values,'period':period,
             'dur':duration,'fs':fs,'file':file,'samples':samples}
        print('done')
        return sig
    


# if __name__ == '__main__':
#     freeze_support()
    

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
        




class ResMeas:
    def __init__(self,fl=None,ft=None,ff=None):
        if fl is not None:
            self.fl=fl
        
        if ft is not None:
            self.ft=ft

        if ff is not None:
            self.ff=ff
        pass
    
    def __get_sig_stats__(self,sig):
        seq=sig['signal']
        N = len(seq)
        peak = np.max(np.abs(seq))
        mean = np.mean(seq)
        std= np.std(seq)
        
        m2 = std**2
        m3 = np.sum((seq - mean)**3) / N
        m4 = np.sum((seq - mean)**4) / N
        mad = np.mean(np.abs(seq - mean))
        rms = np.sqrt(np.sum(seq**2)) / N
        
        kurt = m4 / m2**2
        kurt_rel = kurtosis(seq) # relativni spicatost
        skew = m3 / m2**1.5 # sikmost
        
        clearance_factor = peak / (np.sum(np.sqrt(np.abs(seq))) / N)**2
        crest_factor = peak / rms
        impulse_factor = peak / (np.sum(np.abs(seq)) / N)
        
       
        
        sig_param={
            "peak": peak,
            "mean": mean,
            "std": std,
            "mad": mad,
            "rms": rms,
            "kurt": kurt,
            "kurt_rel": kurt_rel,
            "skew": skew,
            "clearance_factor": clearance_factor,
            "crest_factor": crest_factor,
            "impulse_factor": impulse_factor}
        
        sig['sig_param']=sig_param
        
        return sig
    
    def __findom__(self):
        spdfy=self.spectrum
        ff=0
        ft=0
        fl=0
        maxf=0
        
        
        
        for index,row in spdfy.iterrows():
            if (row['ftype']=='ff') & (ff==0):
                maxy=spdfy[spdfy['ftype']=='fl']['yf'].values.max()
                if (row['xf']>500) & (row['yf']>maxy*0.7) :
                    ff=row['xf']
                    
                    maxf=ff*2.5
                    self.ff['dom_freq_id']=row['peaks']
                

            if (row['ftype']=='ft') & (ft==0) & (ff!=0):
                if row['xf']>(ff+200):
                   ft=row['xf']
                   self.ft['dom_freq_id']=row['peaks']
                
            if (row['ftype']=='fl') & (fl==0) & (ft!=0):
                maxy=spdfy[(spdfy['ftype']=='fl') & (spdfy['xf']>row['xf'])]['yf'].values.max()
                if (row['xf']>(ff+500)) & (row['yf']>maxy*0.85):
                    fl=row['xf']
                    self.fl['dom_freq_id']=row['peaks']
                
            if (row['xf']>maxf) & (fl!=0) & (ff!=0) & (ft!=0):
                break
            
    def GetDom(self):
        dom=dict()
        for key,fs in zip(['ff','ft','fl'],[self.ff,self.ft,self.fl]):
            xf=fs['spectrum']['xf']
            yf=fs['spectrum']['yf']
            idx=fs['dom_freq_id']
            amp="{:s}_amp".format(key)
            dom[key]=xf[idx]
            dom[amp]=yf[idx]
        
        return dom
    
    
    def GetSpectrum(self):
        
        df=pd.DataFrame()
        n=0
        for sig, ftype in zip([self.fl,self.ft,self.ff],['fl','ft','ff']):
            n+=1
            sig=self.__get_sig_stats__(sig)
            
            xf,yf=self.__get_fft__(sig)
            
            freq_diff=yf.max()-yf.min()
            
            peaks, properties = find_peaks(yf, distance=50,prominence=(freq_diff*0.01, freq_diff))
            # peaks, properties = find_peaks(yf, distance=50,prominence=(freq_diff*0.05, freq_diff))
            
            spec={"xf":xf,"yf":yf,"peaks":peaks,"properties":properties,"ff":xf[peaks]}
            sig['spectrum']=spec
            
            
            speca=pd.DataFrame({'xf':xf[peaks],'yf':yf[peaks],'peaks':peaks})
            speca['ftype']=ftype    
            speca['spec_id']=n
            
            df=pd.concat([df,speca],axis=0)
            
        
        spdfy=df
        spdfy=spdfy.sort_values(by='xf',ascending=True)
        spdfy=spdfy.reset_index()
        spdfy=spdfy[spdfy['xf']>200]
        self.spectrum=spdfy
        self.__findom__()
        pass
    
    
    
    def __get_fft__(self,sig):
        signal=sig['signal']
        n = 4096
        ps = abs(np.fft.rfft(signal * get_window('hamming', signal.size)))

        b, a = butter(8, 0.125)
        ps = filtfilt(b, a, ps, padlen=150)

        time_step=1/sig['fs']
        freqs = np.linspace(0,sig['fs']/2,ps.size)

        ps=20*np.log10(np.abs(ps))
        idx=(freqs>0) & (freqs<sig['fs']/2*0.9)
        
        x_freq=freqs[idx]
        y_ps=ps[idx]
        return x_freq,y_ps
        
        
    

class Signal:
    def __init__(self,fs=None,period=None,duration=None,signal=None):
        
        if fs is not None:
            self.fs=fs
            self.period=1/self.fs
        pass
    
    def ReadSignal(self,filename):
        pass
    
    

class IESignal(Signal):
    
    def __init__(self, folder):
        self.Folder=folder
        
        self.TargetSuffix='.csv'
        
        self.Height=0.
        self.Prominence=4000
        self.Distance=150
        self.FrequencyRange=[4000,6000]
        self.Threshold = 3
        self.Quiet = 9000
        self.PeakDistance = 25
        
        self.ColNames=('Time','Hammer','Sensor')
        
    
    def SetFreqRange(self,rang):
        self.FrequencyRange=rang
        
    def ReadSignals(self):
        
        # colnames=('Time','Hammer','Sensor') # time[s], voltage at sensor and hammer [V]
        
        if self.FileList.shape[0]>0:
            for i in range(self.SignalCount):
                self.i = i
            # for i in range(99):
                df = pd.read_csv (self.FileList.filename[i],header=9,delimiter=';',names=self.ColNames)
                self.Signal=df
                # print(len(self.Signal))
                
                if i==0:
                    result=self.stats(self.Signal.Sensor,i)
                    fresult=self.find_resonance(self.Signal,i)
                    # dresult = self.ham_sen_delay(i)
                    dresult = self.Delay(i)
                    
                else:
                    statparameters=self.stats(self.Signal.Sensor,i)
                    fparam=self.find_resonance(self.Signal,i)
                    # dparam = self.ham_sen_delay(i)
                    dparam = self.Delay(i)
                    
                    result=pd.concat([result,statparameters])
                    fresult=pd.concat([fresult,fparam])
                    dresult=pd.concat([dresult,dparam])
                    
                # statparameters.index=i
                self.Param=pd.concat([result,fresult],axis=1)
                self.Param=pd.concat([self.Param,dresult],axis=1)
                print('Done ',i+1,'/',self.SignalCount)
                # print('My name is', os.getlogin(), 'and I am', 42)
            if self.Param.shape[0]==self.SignalCount:
                self.OutTable=pd.concat([self.FileList,self.Param],axis=1)
   
    def UpdateFeatures(self):
            
        pass
    
    def SaveFeatures(self):
        if len(self.OutTable)>0:
            self.OutTable.to_csv('features.csv')
            print("Exported {} observations".format(self.OutTable.shape[0]))
        else:
            print('No features to export')
        # pass
    
    def LoadFeatures(self, csv_file):
        tmp=pd.read_csv (csv_file)
        if tmp.shape[0]>0:
            self.OutTable=tmp
            self.SignalCount=self.OutTable.shape[0]
        # pass
        
        
    def MakeFileList(self):
        
        FileList=pd.DataFrame(columns=("filename","rec","sample","T", "burnt","mode","device"))
        row=0
        for root,dir,files, in os.walk(self.Folder,topdown=True):
            # allfiles=files
            
            if len(files)>0:
                for i in range(len(files)):
                    if files[i].endswith(self.TargetSuffix):
                        
                        filename=root + '\\' + files[i]
                        
                        prsedinfo=self.parse_info(filename,row)
                        row=row+1
                        FileList=pd.concat([FileList,prsedinfo])
        self.FileList=FileList
        self.SignalCount=row
            
        
    def S_23_8275_Make_File_List(self):
      FileList=pd.DataFrame(columns=("filename", "slab", "bind", "measurement", "date_and-time"))
      # FileList["filename"]= None
      row=0
      for root,dir,files, in os.walk(self.Folder,topdown=True):
          # allfiles=files
          
          if len(files)>0:
              for i in range(len(files)):
                  if files[i].endswith(self.TargetSuffix):
                      
                      filename=root + '\\' + files[i]
                      
                      prsedinfo=self.S_23_8275_parse_info(filename, row)
                      row=row+1
                      FileList=pd.concat([FileList,prsedinfo])
      self.FileList=FileList
      self.SignalCount=row  


    @staticmethod
    def parse_info(filename, index):
        
        x = re.search(r"[Rr]eceptura( *|_*)[RAB]", filename).group()
        rec = re.search(r"[RAB]$", x).group()
        
        x = re.search(r"\d+( *|_*)°C|REF", filename).group()
        if x == "REF":
            T = "20"
        else:
            T = re.search(r"^\d*\d", x).group()
        T = int(T)
        
        
        x = re.search(r"[Pp]o( *|_*)[Vv][yý]palu", filename)
        if x is None:
            burnt = False
        elif x is not None:
            burnt = True
            
        x = re.search(r"UB[01]", filename).group()
        mode = x
        
        x = re.search(rec+r"\d+", filename).group()
        sample = re.search(r"\d+", x).group()
        sample = int(sample)
        
        x = re.search(r"(Piezo( |_)(Gel|Vosk))|Mikrofon", filename).group()
        device = x
        
        x = re.search(r"( |_)(2|3|4|5)", filename)
        if x is None:
            measurement = "1"
        else:
            measurement = re.search(r"(2|3|4|5)", x.group()).group()
        measurement = int(measurement)
        
        result=pd.DataFrame({
            "filename": filename,
            "rec": rec,
            "sample": sample, 
            "T": T, 
            "burnt": burnt,
            "mode": mode, 
            "device": device,
            "measurement": measurement
            },index=[index])
        
        return result

    @staticmethod
    def S_23_8275_parse_info(filename, index):
        x = re.search(r"[Dd]eska( *|_*)[BC]( *|_*)\d+.*\d", filename).group()
        slab = re.search(r"[BC]( *|_*)\d+.*\d", x).group()
        
        x = re.search(r"[Gg]el|[Pp]lastelina", filename).group()
        bind = x
        
        x = re.search(r"[Bb]od( *|_*|-*)\d+( *|_*|-*)\d+( *|_*|-*)\d+( *|_*|-*)\d+", filename).group()
        measurement = re.search(r"\d+", x).group()
        measurement = int(measurement)
        date_and_time = re.search(r"2023\d+( *|_*|-*)\d+( *|_*|-*)\d+", x).group()
        
        result = pd.DataFrame({
            "filename": filename,
            "slab": slab,
            "bind": bind,
            "measurement": measurement,
            "date_and_time": date_and_time
            },index=[index])
        
        return result
        

    @staticmethod
    def MyFFT(signal,freq):
        """
        vezme array mereni signalu a vzorkovaci frekvenci
        da FT signalu ve frekvencnim spektru
        """
    
        samples=signal.size
    
    
        ywf = fft(signal)
        yf=2.0/samples * np.abs(ywf[0:samples//2])
    
        xf = np.linspace(0, freq/2, round(samples/2))
        
        return xf,yf
    
    
    @staticmethod
    def stats(seq,index):
        N = len(seq)
        peak = np.max(np.abs(seq))
        mean = np.mean(seq)
        std= np.std(seq)
        
        m2 = std**2
        m3 = np.sum((seq - mean)**3) / N
        m4 = np.sum((seq - mean)**4) / N
        mad = np.mean(np.abs(seq - mean))
        rms = np.sqrt(np.sum(seq**2)) / N
        
        kurt = m4 / m2**2
        kurt_rel = kurtosis(seq) # relativni spicatost
        skew = m3 / m2**1.5 # sikmost
        
        clearance_factor = peak / (np.sum(np.sqrt(np.abs(seq))) / N)**2
        crest_factor = peak / rms
        impulse_factor = peak / (np.sum(np.abs(seq)) / N)
        
       
        
        result=pd.DataFrame({
            "peak": peak,
            "mean": mean,
            "std": std,
            "mad": mad,
            "rms": rms,
            "kurt": kurt,
            "kurt_rel": kurt_rel,
            "skew": skew,
            "clearance_factor": clearance_factor,
            "crest_factor": crest_factor,
            "impulse_factor": impulse_factor
            },index=[index])
        
        return result

    
    def find_resonance(self,signal,idx2):
        freq= 1/(signal.Time[1]- signal.Time[0])
        f,y= self.MyFFT(signal.Sensor.to_numpy(),freq)
        
        if len(self.FrequencyRange)==2:
            idx=(f>self.FrequencyRange[0]) & (f<self.FrequencyRange[1])
            f=f[idx]
            y=y[idx]
            
        ay=y*1e+3
        
        
        maxamp=max(ay[10:-1])
        peaks, properties = find_peaks(ay, height=maxamp*0.2,prominence=(None, self.Prominence),distance=self.Distance)
    
        idx=np.linspace(0,peaks.size-1,peaks.size)
        
        peaksexport=pd.DataFrame.from_dict(properties)

        
        
        
        xmin=properties["left_bases"]
        xmax=properties["right_bases"]
        
        # width=f[peaksexport['right_bases']]-f[peaksexport['left_bases']]
        width=xmax-xmin
        
        umax=y[peaks]
        trsh=umax/np.power(2,0.5)
        
        lid=[]
        rid=[]
            
        for i in range(len(peaks)):
            
            for n in range(peaks[i]):
                nt=peaks[i]-n
                if y[nt]<trsh[i]:
                    lid.append(f[nt])
                    break
                
            for n in range(peaks[i]):
                nt=peaks[i]+n
                if y[nt]<trsh[i]:
                    rid.append(f[nt])
                    break
        
        fl=np.array(lid)
        fr=np.array(rid)
        fd=np.array(f[peaks])
        if (fr.size==peaks.size) & (fl.size==peaks.size) & (fd.size==peaks.size):
            attenuation=(fr-fl)/fd*math.pi
        else:
            attenuation=np.nan
    
                    
        
        # attenuation=np.log(y[peaks]/width)
        peaksdf=pd.DataFrame({"indexes": peaks, "frequency": f[peaks],"orgamp": y[peaks],"width": width, "attenuation": attenuation})
    
        result = pd.concat([peaksdf,peaksexport],axis=1)
        
        
        r2=result.sort_values(by=['orgamp'],ascending=False)
        r2.reset_index()
        r3=r2[r2.index==0]
        r3.index=[idx2]
        #r3=r3.reset_index()
        # r2.loc(range(1,r2.shape[0]),axis=0,inplace=True)
        # r2[0].index=idx

        return r3
    
    


    def first_sig(self, seq):
        """
        zjisti prvni bod, ktery zaznamenal signal 
        odchyleny od sumu o thr*std (stanoveny prvnimi qui body, ktere by mely byt klid)
        a ktery je nasledovan aspon 3 dalsimi odchylenymi body 
        """
        # silence = np.array( seq[:self.Quiet])
        silence = seq[:self.Quiet]
        mean, std = self.stats(silence, 0)["mean"][0], self.stats(silence, 0)["std"][0]
        
        for i in range(self.Quiet, len(seq) - 3):
            if np.abs( seq[i] - mean ) > self.Threshold*std :
                if np.abs( seq[i+1] - mean ) > self.Threshold*std :
                    if np.abs( seq[i+2] - mean ) > self.Threshold*std :
                        if np.abs( seq[i+3] - mean ) > self.Threshold*std :
                            return i
    
    def FirstSig(self, seq):
        """
        zjisti prvni bod, jehoz hodnota je aspon (Threshold) maxima signalu
        """
        peak = self.stats(seq, 0)["peak"][0]
        print(f"Peak: {peak}")
        
        if self.Threshold > 1:
            raise ValueError
        
        for i, pt in enumerate(seq):
            if pt > self.Threshold * peak:
                print(f"i_first: {i}")
                return i
        
        return None
    
    def SignalStart(self, seq):
        dt= self.Signal.Time[1] - self.Signal.Time[0]
        f = self.find_resonance(self.Signal, self.i)["frequency"][self.i]
        T = 1/f
        i_per_T = int(T/dt)
        
        i_max = np.where(seq == np.max(seq))[0][0]
        i_min = np.where(seq == np.min(seq))[0][0]

        if(i_max < i_min):
            i_peak = i_max
        elif(i_min < i_max):
            i_peak = i_min
        peak = seq[i_peak]
        
        U_start, i_start = peak, i_peak
        mean = self.stats(seq[:i_peak-1000], 0)["mean"][0]
        std = self.stats(seq[:i_peak-1000], 0)["std"][0]
        
        while(\
              np.abs(seq[i_start] - mean) > self.Threshold*std \
          and np.abs(seq[i_start - int(i_per_T/4)] - mean) > self.Threshold*std\
          and np.abs(seq[i_start - int(i_per_T*3/4)] - mean) > self.Threshold*std\
          ):
            i_start -= 1
            # U_start = seq[i_start]
            # mean = self.stats(seq[:i_start], 0)["mean"][0]
            # std = self.stats(seq[:i_start], 0)["std"][0]
        
        return i_start
    
    # def FirstPeak(self,seq):
    #     pass
    #     mean = self.stats(seq[:self.Quiet], 0)["mean"][0]
    #     std = self.stats(seq[:self.Quiet], 0)["std"][0]
    #     peaks = find_peaks(seq, distance = 20, plateau_size = 1)
        
    #     return peaks[0]
        
    def ham_sen_delay(self, index):
        
        # ham_start = self.first_sig(self.Signal.Hammer)
        # sen_start = self.first_sig(self.Signal.Sensor)
        
        # ham_start = self.FirstSig(self.Signal.Hammer)
        # sen_start = self.FirstSig(self.Signal.Sensor)
        
        ham_start = self.SignalStart(self.Signal.Hammer)
        sen_start = self.SignalStart(self.Signal.Sensor)
        
        # ham_peak = self.FirstPeak(self.Signal.Hammer)
        # sen_peak = self.FirstPeak(self.Signal.Sensor)

        delay = None
        
        if sen_start is not None and ham_start is not None:
            delay = self.Signal.Time[sen_start] - self.Signal.Time[ham_start]
       
        print(f"delay: {delay}")
        return pd.DataFrame({"delay": delay}, index = [index])
    
    def Delay(self, index):
        delay = None
        
        sensor_mean = self.stats(self.Signal.Sensor[:self.Quiet], 0)["mean"][0]
        sensor_std = self.stats(self.Signal.Sensor[:self.Quiet], 0)["std"][0]    
        # hammer_mean = self.stats(self.Signal.Hammer[:self.Quiet], 0)["mean"][0]    
        # hammer_std = self.stats(self.Signal.Hammer[:self.Quiet], 0)["std"][0]    
        
        try:    
            i_hammer_min = np.where(self.Signal.Hammer == np.min(self.Signal.Hammer))[0][0]
            hammer_strike = self.Signal.Time[i_hammer_min]
        
            i_sensor_peaks = find_peaks(\
                                    self.Signal.Sensor, \
                                    distance = self.PeakDistance,\
                                    height = sensor_mean + self.Threshold*sensor_std)
            peak_times = self.Signal.Time[i_sensor_peaks]
            first_peak = peak_times[peak_times > hammer_strike][0]
        
        
            delay = first_peak - hammer_strike
            print(f"delay: {delay}")
            return pd.DataFrame({"delay": delay}, index = [index])
        except Exception:
            return self.ham_sen_delay(index)
    
    def ShowPlot(self,sigid:int=None,draw:bool=None):
        
        if sigid is None:
            sigid=0
            
        # colnames=('Time','Hammer','Sensor')
        df = pd.read_csv (self.FileList.filename[sigid],header=9,delimiter=';',names=self.ColNames)
        
        
        self.Signal=df
        
        freq= 1/(self.Signal.Time[1]- self.Signal.Time[0])
        
        f,y= self.MyFFT(self.Signal.Sensor.to_numpy(),freq)
        
        if len(self.FrequencyRange)==2:
            idx=(f>self.FrequencyRange[0]) & (f<self.FrequencyRange[1])
            f=f[idx]
            y=y[idx]
        
        if draw is not None:
            if draw is True:
                fig, axs = plt.subplots(2)
                axs[0].plot(self.Signal.Time,self.Signal.Sensor)
                axs[1].plot(f,y)

        result={
            "s_time": (self.Signal.Time.to_numpy()),
            "s_amp": (self.Signal.Sensor.to_numpy()),
            "s_h_amp": (self.Signal.Hammer.to_numpy()),
            "f_freq": (f),
            "f_amp": (y),
            }
        return result




class DBComm:
    def __init__(self,filename=None):
        self.Dir=os.getcwd()
        if filename is None:
            self.Filename="IEStorage.db"
            self.DBPath="%s\%s" %(self.Dir,self.Filename)    
        else:
            if os.path.isabs(filename):
                self.DBPath=filename
            else:
                self.Filename=filename
                self.DBPath="%s\%s" %(self.Dir,self.Filename)  
            
        self.DBExist=False

    
    def LoadDBPath(self,inifile=None):

        if inifile is None:
            inifile=r"%s\\%s" %(os.getcwd(),"inifile.txt")

        if os.path.isfile(inifile):
            try:
                sett=dict()
                with open(inifile, 'w') as f: 
                    for key, value in sett.items(): 
                        f.write('%s:%s\n' % (key, value))
                
                self.DBPath=sett["dbfile"]
                self.StartDB()
                return False
                
            except:
                print("Wrong datatype for inifile!")
                return True
        else:
            print("File doesnt exist")
            return True
    
    
    def saveDBPath(self,filename=None):
        if filename is None:
            filename=r"%s\\%s" %(os.getcwd(),"inifile.txt")
            
        sett={"dbfile":self.DBPath}
        with open(filename, 'w') as convert_file:
             convert_file.write(json.dumps(sett))
    
    def StartDB(self):
        if os.path.isfile(self.DBPath):
            #exist ok
            self.Conn = sqlite3.connect(self.DBPath)
            print("Succesfully connected to '%s'" % self.DBPath)
            self.DBExist=True
            
        else:
            #doesnt exsit, must create
            print("Database '%s' created succesfully" % self.DBPath)
            self.Conn = sqlite3.connect(self.DBPath)
            self.Conn.execute('''CREATE TABLE CONTENT
                     (ID INTEGER PRIMARY KEY AUTOINCREMENT,
                     TABNAME       TEXT     NOT NULL);''')
                     
            self.Conn.execute('''CREATE TABLE SIGNALS
                     (ID INTEGER PRIMARY KEY AUTOINCREMENT,
                     DATETIME       TEXT    NOT NULL,
                     LABEL          TEXT    NOT NULL,
                     FLSIGNAL       BLOB    NOT NULL,
                     FFSIGNAL       BLOB    NOT NULL
                     FTSIGNAL       BLOB    NOT NULL);''')
                     
                     
            self.Conn.execute('''CREATE TABLE PARAMS
                      (ID INTEGER PRIMARY KEY AUTOINCREMENT,
                      FIELD       TEXT     NOT NULL,
                      VALUE       TEXT     NOT NULL);''')
            
            self.Conn.commit()
            
            sqlite_insert_blob_query = """INSERT INTO PARAMS
                                      (FIELD, VALUE) VALUES ('VERSION', '1.0')"""
            self.Conn.execute(sqlite_insert_blob_query)
            
            self.Conn.commit()                     
            self.Conn.close()
            self.DBExist=True
            
        self.saveDBPath()
        
            
            
    def testDB(self,filename=None):
        if filename is None:
            filename=self.DBPath
        
        try:
            self.Conn = sqlite3.connect(filename)
            cursor = self.Conn.cursor()
            qStr = """SELECT VALUE from PARAMS where FIELD ='VERSION' """
            cursor.execute(qStr)
            record = cursor.fetchall()
            if record==0:
                print("Wrong DB")
                return False
            else:
                print("Right DB")
                return True
                
            self.Conn.close()
        except:
            print("Cannot connect")
            return False
                
        
        
    def AddSignal(self,data):
        if self.DBExist==False:
            self.StartDB()
            
        self.Conn = sqlite3.connect(self.DBPath)
        cursor = self.Conn.cursor()

        sqlite_insert_blob_query = """INSERT INTO SIGNALS
                                  (datetime, data) VALUES (?, ?)"""

        dataC=data.copy()
        
        # xi = np.array(dataC["time"], dtype='<f')
        yi = np.array(dataC["signal"], dtype='<f')
        # dataC["time"]=xi.tobytes()
        dataC["signal"]=yi.tobytes()        


        pdata = pickle.dumps(dataC, pickle.HIGHEST_PROTOCOL)
        date=dataC["datetime"]
        
        strDate=date.strftime("%Y%m%d-%H%M%S")
        
        tup=(strDate,pdata)
        
        cursor.execute(sqlite_insert_blob_query, tup)
        self.Conn.commit()
        self.Conn.close()
        
    def GetSignalTable(self):
        if self.DBExist==False:
            self.StartDB()
            
        self.Conn = sqlite3.connect(self.DBPath)
        cursor = self.Conn.cursor()

        qStr = """SELECT ID, DATETIME from SIGNALS where id > 0 """
        # cursor.execute(sql_fetch_blob_query)

        self.SignalTable = pd.read_sql_query(qStr, self.Conn)

        record = cursor.fetchall()
        
        if record==0:
            self.SignalTable=pd.DataFrame(columns=["ID","DATETIME"])
            
        self.Conn.close()
        
        return self.SignalTable
        
    def GetSignal(self,ID):
        if self.DBExist==False:
            self.StartDB()
        
        
        if self.__ExistSignal(ID)==True:
            self.Conn = sqlite3.connect(self.DBPath)
            cursor = self.Conn.cursor()
            qStr = """SELECT * from SIGNALS where id = %d """  % ID
            
            df = pd.read_sql_query(qStr, self.Conn)
            
            dicti=pickle.loads(df["DATA"][0])
            # dicti["time"]=np.frombuffer(dicti["time"], dtype='<f') 
            dicti["signal"]=np.frombuffer(dicti["signal"], dtype='<f') 
            
            # df["DATA"][0]=dicti
            
            self.Signal=dicti
            
            self.Conn.close()

            return dicti
        else:
            print("Desired signal with ID:%d doesnt exist" % ID)
             
             
    def __ExistSignal(self,ID):
        self.Conn = sqlite3.connect(self.DBPath)
        cursor = self.Conn.cursor()
        cursor.execute("SELECT count(*) FROM SIGNALS WHERE ID = ?", (ID,))
        data=cursor.fetchone()[0]
        if data==0:
            return True
        else:
            return False
        self.Conn.close()
        
    def __del__(self):
        
        self.Client.close()
        print("Closing IEControl")
