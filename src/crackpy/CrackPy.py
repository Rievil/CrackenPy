# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:20:30 2023

@author: dvorr
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

from PIL import Image as PImage
import cv2
import albumentations as A

import time
import os
from tqdm.notebook import tqdm

# !pip install -q segmentation-models-pytorch
# !pip install -q torchsummary

from torchsummary import summary
import segmentation_models_pytorch as smp
from wand.image import Image as WI
from skimage.morphology import medial_axis, skeletonize

from skimage.measure import label, regionprops, regionprops_table
import math
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import pdist

class CrackPy:
    def __init__(self):
        self.impath=''
        self.is_cuda=torch.cuda.is_available()
        self.device = torch.device("cuda" if self.is_cuda else "cpu")
        
        self.img_channels=3
        self.encoder_depth=5
        self.class_num=5
        
        self.model_type='resnext101_32x8d'
        self.model_path=r"Models\resnext101_32x8d_N387_C5_30102023.pt"
        
        self.model = smp.FPN(self.model_type, in_channels=self.img_channels,
                             classes=self.class_num,activation=None, encoder_depth=self.encoder_depth)  
        self.model_path=self.model_path
        self.__loadmodel__()
        self.reg_props=('area','centroid','orientation','axis_major_length','axis_minor_length')
        self.pred_mean=[0.485, 0.456, 0.406]
        self.pred_std=[0.229, 0.224, 0.225]
        self.patch_size=416
        self.crop=False
        self.pixel_mm_ratio=1
        
        pass
    
    def __loadmodel__(self):
        if self.is_cuda==True:
            self.model.load_state_dict(torch.load(self.model_path))
        else:
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        self.model.eval()
        
    def __ReadImg__(self,impath):
        self.impath=impath
        if '.heic' in impath.lower():
          img=WI(filename=impath)
          img.format='jpg'
          img_buffer=np.asarray(bytearray(img.make_blob()), dtype=np.uint8)
          img = cv2.imdecode(img_buffer, cv2.IMREAD_UNCHANGED)
        else:
          img = cv2.imread(self.impath)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    def ClassifyImg(self,impath):
        self.impath=impath
        img = cv2.imread(self.impath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img=cv2.resize(img,(416, 416), interpolation=cv2.INTER_NEAREST)
        img = PImage.fromarray(img)
        self.img=img
        self.mask=self.__predict_image__(self.img)
        return self.mask
        
    def GetImg(self,impath):
        if impath is not self.impath:
            self.impath=impath
            img = cv2.imread(self.impath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img=cv2.resize(img,(416, 416), interpolation=cv2.INTER_NEAREST)
            # img = PImage.fromarray(img)
            self.img=img
            mask=self.IterateMask()
        return mask
    
    def __del__(self):
        torch.cuda.empty_cache()
        
    def GetRatio(self,length=None,width=None):
        reg_props=('area','centroid','orientation','axis_major_length','axis_minor_length')

        if length is None:
            self.length=160
        else:
            self.length=length
            
        if width is None:
            self.width=40
        else: 
            self.width=width

        mask = np.array(self.mask)
        bw_mask=mask[:,:]==1
        image=bw_mask.astype(np.uint8)
        label_img = label(image)

        props_mat = regionprops_table(label_img, properties=reg_props)
        dfmat=pd.DataFrame(props_mat)
        dfmat.sort_values(by=['area'],ascending=False)
        dfmat=dfmat.reset_index()


        l_rat=self.length/dfmat['axis_major_length'][0]
        w_rat=self.width/dfmat['axis_minor_length'][0]
        m_rat=(l_rat+w_rat)/2
        self.pixel_mm_ratio=m_rat
        return self.pixel_mm_ratio
        


    def __predict_image__(self,image):
        self.model.eval()
        t = T.Compose([T.ToTensor(), T.Normalize(self.pred_mean, self.pred_std)])
        image = t(image)
        self.model.to(self.device); image=image.to(self.device)
        with torch.no_grad():
            image = image.unsqueeze(0)
            output = self.model(image)

            masked = torch.argmax(output, dim=1)
            masked = masked.cpu().squeeze(0)
        return masked
    
    def MeasureBW(self):
        back_bw=self.mask[:,:]==0
        back_bw=back_bw.astype(np.uint8)
        
        mat_bwo=self.mask[:,:]==1
        mat_bwo=mat_bwo.astype(np.uint8)
        
        kernel = np.ones((50, 50), np.uint8) 
        mat_bw = cv2.dilate(mat_bwo, kernel, iterations=1)
        mat_bw = cv2.erode(mat_bw, kernel) 

        
        crack_bw=self.mask[:,:]==2
        crack_bw=crack_bw.astype(np.uint8)
        
        
        
        pore_bw=self.mask[:,:]==3
        pore_bw=pore_bw.astype(np.uint8)
        
        crack_bw = cv2.bitwise_and(mat_bw,crack_bw)
        pore_bw = cv2.bitwise_and(mat_bw,pore_bw)
        
        self.masks={'back':back_bw,'mat':mat_bwo,'crack':crack_bw,
                    'pore':pore_bw}
        
            
        total_area=back_bw.shape[0]*back_bw.shape[1]
        back_area=back_bw.sum()
        spec_area=total_area-back_area
        crack_area=crack_bw.sum()
        pore_area=pore_bw.sum()
        
        mat_area=total_area-(crack_area+spec_area+pore_area)

        crack_ratio=crack_area/spec_area
        
        skel = skeletonize(crack_bw, method='lee')
        
        crack_length=skel.sum()
        crack_avg_thi=crack_area/crack_length
        
        result={"spec_area":spec_area*self.pixel_mm_ratio,
                "mat_area":mat_area*self.pixel_mm_ratio,
                "crack_area":crack_area*self.pixel_mm_ratio,
                "crack_ratio":crack_ratio,
                "crack_length":crack_length*self.pixel_mm_ratio,
                "crack_thickness":crack_avg_thi*self.pixel_mm_ratio,
                "pore_area":pore_area*self.pixel_mm_ratio}
        
        self.bw_stats=result
        self.__MeasurePores__()
        return result
    
    def __MeasurePores__(self):
        image_pore=self.masks['pore']
        label_img_pore = label(image_pore)
        
        props_pore = regionprops_table(label_img_pore, properties=self.reg_props)
        dfpores=pd.DataFrame(props_pore)
        
        mask = dfpores['area']<10
        dfpores = dfpores[~mask]
        
        dfpores.sort_values(by=['area'],ascending=False)
        dfpores=dfpores.reset_index()
        
        points = np.array([dfpores['centroid-1'],dfpores['centroid-0']])
        points=np.rot90(points)
        arr=pdist(points,metric='minkowski')
        
        avgdist=arr.mean()
        area=dfpores['area'].mean()
        self.bw_stats['avg_pore_distance']=avgdist
        self.bw_stats['avg_pore_size']=area

    def SetCropDim(self,dim):
        self.crop_rec=dim
        self.crop=True
    
    def IterateMask(self):
        
            
        imgo = self.img
        sz=imgo.shape
        step_size=self.patch_size
        
        xcount=sz[0]/step_size
        xcount_r=np.ceil(xcount)
        ycount=sz[1]/step_size
        ycount_r=np.ceil(ycount)
        

        blank_image = np.zeros((int(sz[0]),int(sz[1])), np.uint8)
        
        
        width=step_size
        height=width
        
        
        for xi in range(0,int(xcount_r)):
            for yi in range(0,int(ycount_r)):
                
                if xi<xcount-1:
                    xstart=width*xi
                    xstop=xstart+width
                else:
                    xstop=sz[0]
                    xstart=xstop-step_size
                
                if yi<ycount-1:
                    ystart=height*yi
                    ystop=ystart+height
                else:
                    ystop=sz[1]
                    ystart=ystop-step_size
                    
                
                cropped_image = imgo[xstart:xstop, ystart:ystop]
                
                
                mask=self.__predict_image__(cropped_image)
                blank_image[xstart:xstop, ystart:ystop]=mask
                
        if self.crop==True:
            dim=self.crop_rec
            imgo=self.img[dim[0]:dim[1],dim[2]:dim[3]]
            self.img_crop=imgo
            self.mask=blank_image[dim[0]:dim[1],dim[2]:dim[3]]
        else:        
            self.mask=blank_image
            
        return self.mask


        