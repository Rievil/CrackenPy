# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 11:45:19 2023

@author: dvorr
"""
import pandas as pd
import os
import datetime
from scipy import io
import numpy as np
from tqdm import trange

import os
from CrackPy import CrackPy as CrackPy
from matplotlib import pyplot as plt

class CrackManager:
    def __init__(self,folder=None):
        self.crack_py=CrackPy()
        self.impick_set=False
        
        if folder is None:
            pass
            self.folder=''
        else:
            self.folder=folder
            self.__listdir__()
            
    def __listdir__(self):
        im_path=self.folder
        df=pd.DataFrame()
        for root, dirs, files in os.walk(im_path,topdown=False):
           for name in files:
               file_path   = root + '/' + name
               created     = os.path.getctime(file_path)
               modified    = os.path.getmtime(file_path)
               
        
               date=datetime.datetime.fromtimestamp(modified)
               dfi=pd.DataFrame({"folder":root,"name":name,"modified":modified,"date":date},index=[0])
               df=pd.concat([df,dfi],axis=0)
        
        df['order']=df['modified']-df['modified'].min()
        df=df.sort_values(by='order', ascending=True)
        df=df.reset_index()
        
        extension = df['name'].str.split('.', expand=True)
        idx=extension[1]=='png'
        
        dff=df[idx]
        self.img_list=dff
    
    
    def SetPath(self,path):
        self.folder=path
        self.__listdir__()
        self.impick_set=False
        
    def SetCropDim(self,dim):
        self.crack_py.SetCropDim(dim)
        
    def SetIdx(self,idx=None,count=None):
        self.__listdir__()
        
        if idx is None:
            idxf=np.linspace(0,self.img_list.shape[0],count)
            self.impick_idx = idxf.astype(int)
        
        if count is None:
            self.impick_idx=idx

        self.impick_set=True
    
    def Scan(self):
        self.__listdir__()
        return self.__processimgs__()
    
    def Show(self,row=None):
        if row is None:
            self.curr_img=0
        else:
            self.curr_img=row
        
        file_name='{:s}\{:s}'.format(self.img_list['folder'][self.curr_img],self.img_list['name'][self.curr_img])
        self.crack_py.GetImg(file_name)
        fig,(ax2)=plt.subplots(1,1)
        
        # ax1.imshow(self.crack_py.img)
        if self.crack_py.crop==True:
            ax2.imshow(self.crack_py.img_crop)
            c=ax2.imshow(self.crack_py.mask,alpha=0.8,cmap='jet')
        else:
            ax2.imshow(self.crack_py.img)
            c=ax2.imshow(self.crack_py.mask,alpha=0.8,cmap='jet')
            
        plt.axis("off")
        cbar=fig.colorbar(c, ax=ax2,ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(['Back', 'Matrix', 'Cracks','Pores']) 
        plt.tight_layout()
    
    # def __del__(self):
        # self.crack_py.delete()

    def __processimgs__(self):
        
        dfr=pd.DataFrame()
        
        if self.impick_set==False:
            self.impick_idx=self.img_list.index

        for i in trange (0,self.impick_idx.size-1):
            file_name='{:s}\{:s}'.format(self.img_list['folder'][self.impick_idx[i]],self.img_list['name'][self.impick_idx[i]])
            
            mask=self.crack_py.GetImg(file_name)
            if i==0:
                self.crack_py.GetRatio()
                
            dc=self.crack_py.MeasureBW()
            dc['label']=self.img_list['name'][self.impick_idx[i]]
            dc['date']=self.img_list['date'][self.impick_idx[i]]
            
            dfn=pd.DataFrame(dc,index=[i])
            dfr=pd.concat([dfr,dfn],axis=0)
            
        self.result=dfr
        return self.result
    