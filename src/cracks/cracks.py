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
import pkg_resources

class CrackPy:
    def __init__(self,model=0):
        self.impath=''
        self.is_cuda=torch.cuda.is_available()
        self.device = torch.device("cuda" if self.is_cuda else "cpu")
        
        self.img_channels=3
        self.encoder_depth=5
        self.class_num=5
        
        self.model_type='resnext101_32x8d'
        self.models=['resnext101_32x8d_N387_C5_30102023','resnext101_32x8d_N387_C5_310124']
        self.default_model=pkg_resources.resource_filename('models', r'{:s}.pt'.format(self.models[1]))
        
        self.model_path='{}'.format(self.default_model)
        
        
        self.model = smp.FPN(self.model_type, in_channels=self.img_channels,
                             classes=self.class_num,activation=None, encoder_depth=self.encoder_depth) 
        
        # self.model_path=
        self.__loadmodel__()
        self.reg_props=('area','centroid','orientation','axis_major_length','axis_minor_length')
        self.pred_mean=[0.485, 0.456, 0.406]
        self.pred_std=[0.229, 0.224, 0.225]
        self.patch_size=416
        self.crop=False
        self.pixel_mm_ratio=1
        self.mm_ratio_set=False
        
        pass
    
    def __loadmodel__(self):
        if self.is_cuda==True:
            self.model.load_state_dict(torch.load(self.model_path))
        else:
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        self.model.eval()
        
    def __ReadImg__(self):

        if '.heic' in self.impath.lower():
          img=WI(filename=self.impath)
          img.format='jpg'
          img_buffer=np.asarray(bytearray(img.make_blob()), dtype=np.uint8)
          img = cv2.imdecode(img_buffer, cv2.IMREAD_UNCHANGED)
        else:
          img = cv2.imread(self.impath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
        
    def ClassifyImg(self,impath):
        self.impath=impath
        img = cv2.imread(self.impath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img=cv2.resize(img,(416, 416), interpolation=cv2.INTER_NEAREST)
        img = PImage.fromarray(img)
        self.img=img
        self.mask=self.__predict_image__(self.img)
        return self.mask
        
    def GetMask(self,impath=None,img=None):
        self.mm_ratio_set=False
        if impath is not None:
            if impath is not self.impath:
                self.impath=impath
                # img=
                # img = cv2.imread(self.impath)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img=cv2.resize(img,(416, 416), interpolation=cv2.INTER_NEAREST)
                # img = PImage.fromarray(img)
                
                self.img=self.__ReadImg__()
                self.IterateMask()
        elif (impath is None) & (img is not None):
            self.img=PImage.fromarray(img)
            self.IterateMask()
            
        return self.mask
    
    def __del__(self):
        torch.cuda.empty_cache()
        
    def SetRatio(self,length=None,width=None):
        self.mm_ratio_set=True
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
        
        self.orientation = props_mat['orientation']
        
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
    
    
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle
        
class CrackPlot:
    def __init__(self,crackpy):
        self.CP=crackpy
        
    def overlay(self,figsize=[5,4]):
        colors = ["#0027B9", "#0DC9E7", "#E8DD00","#D30101"]
        my_cmap = ListedColormap(colors, name="my_cmap")

        fig,ax=plt.subplots(1,1,figsize=figsize)

        ax = plt.gca()
        ax.imshow(self.CP.img)

        im=ax.imshow(self.CP.mask,alpha=0.7,cmap=my_cmap)


        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
         
        cbar=plt.colorbar(im, cax=cax)
        cbar.set_ticks([0,1,2,3])
        cbar.ax.set_yticklabels(["Back","Matrix","Crack","Pore"])
        cbar.ax.tick_params(labelsize=10,size=0)

        ax.axis("off")
        plt.show()
        
        return fig
    
    def Save(self,fig,name):
        fig.savefig('Plots\{:s}.png'.format(name),dpi=300,bbox_inches = 'tight',
            pad_inches = 0)
        
    def distancemap(self):


        mask=self.CP.mask
        crack_bw=mask[:,:]==2
        crack_bw=crack_bw.astype(np.uint8)

        thresh=crack_bw
        #Determine the distance transform. 
        skel = skeletonize(thresh, method='lee')
        dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5) 
        idx=skel==1
        dist_skel=dist[idx]

          


        fig,(ax1,ax)=plt.subplots(nrows=1,ncols=2,gridspec_kw={'width_ratios': [1, 3]},figsize=(8,5))
        
        ax.imshow(self.CP.img)
        
        if self.CP.mm_ratio_set==True:
            im=ax.imshow(dist*self.CP.pixel_mm_ratio,cmap='jet',alpha=0.8) 
        else:
            im=ax.imshow(dist,cmap='jet',alpha=0.8) 

       

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
         
        cbar=plt.colorbar(im, cax=cax)
        # cbar.set_ticks([0,1,2,3])
        # cbar.ax.set_yticklabels(["Back","Matrix","Crack","Pore"])
        cbar.ax.tick_params(labelsize=10,size=0)

        ax.axis("off")
            
        if self.CP.mm_ratio_set==True:
            arr_dist=dist[skel==1]*2*self.CP.pixel_mm_ratio
            plt.suptitle("Mean thickness {:.2f} mm".format(arr_dist.mean()))
        else:
            arr_dist=dist[skel==1]*2
            plt.suptitle("Mean thickness {:.2f} pixels".format(arr_dist.mean()))
            
        
        ax1.boxplot(arr_dist)
        ax1.get_xaxis().set_ticks([])
        
        plt.show()
        return fig

        