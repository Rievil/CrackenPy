# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 14:48:22 2023

@author: Richard
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

from PIL import Image
import cv2
import albumentations as A

import time
import os
from tqdm.notebook import tqdm

# !pip install -q segmentation-models-pytorch
# !pip install -q torchsummary

from torchsummary import summary
import segmentation_models_pytorch as smp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


# IMAGE_PATH = r'C:\Users\Richard\Vysoké učení technické v Brně\22-02098S - Dokumenty\General\Data\Kamera\DataSets\GACR\GACR_CrackDataset\Images'
# MASK_PATH = r'C:\Users\Richard\Vysoké učení technické v Brně\22-02098S - Dokumenty\General\Data\Kamera\DataSets\GACR\GACR_CrackDataset\Labels'

IMAGE_PATH = r'C:\PyTorchData\GACR_CrackDataset\Images'
MASK_PATH = r'C:\PyTorchData\GACR_CrackDataset\Labels'

#%%

model = smp.FPN('resnext101_32x8d', in_channels=3,classes=5,activation=None, encoder_depth=5)

model.load_state_dict(torch.load(r'Models\resnext101_32x8d_N387_C5.pt'))
model.eval()

#%% 
fig,(ax1,ax2)=plt.subplots(1,2)
#%

img_path=r'C:\PyTorchData\GACR_CrackDataset\Images\ID1_Spec1200_107_Image.png'
# img_path=r'C:\PyTorchData\ConcreteCracksMat\Img\Label_122.png'
# img_path=r'C:\PyTorchData\Test\Img5.png'

def GetImg(impath):
    img = cv2.imread(impath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(416, 416), interpolation=cv2.INTER_NEAREST)
    img = Image.fromarray(img)
    return img

def predict_image(model, image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device); image=image.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        output = model(image)

        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked

img=GetImg(img_path)
ax1.imshow(img)
mask=predict_image(model,img)
ax2.imshow(mask)
plt.savefig("Plots\Example of classification.png")

#%% Enlarging the original image

import numpy as np

fig,(ax1,ax2)=plt.subplots(2,1)
impath=r'C:\Users\Richard\VUT\PomVedi - General\Data\FIJI\6_C_1\ID6_172_Image.png'


imgo = cv2.imread(impath)

imgo = cv2.cvtColor(imgo, cv2.COLOR_BGR2RGB)
ax1.imshow(imgo)
# image = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = 0)

sz=imgo.shape
step_size=416

xcount=sz[0]/step_size
xcount_r=np.ceil(xcount)
ycount=sz[1]/step_size
ycount_r=np.ceil(ycount)


xbo=int(xcount_r*step_size-sz[0])
ybo=int(ycount_r*step_size-sz[1])



# img = cv2.copyMakeBorder(img, 0, xbo, 0, ybo, cv2.BORDER_CONSTANT, None, value = 255)
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
        
        
        mask=predict_image(model,cropped_image)
        blank_image[xstart:xstop, ystart:ystop]=mask


ax2.imshow(blank_image)

#%% Show only selected classes
# fig,(ax1)=plt.subplots(1,1)
# cl_img = np.zeros((int(xcount_r*step_size),int(ycount_r*step_size)), np.uint8)


# plt.imshow(cl_img)
# fig.savefig('TestImg.png')

#%% Skeletonize

import cv2
import numpy as np
 
# img = cv2.imread('TestImg.png',0)

cl_img=np.uint8(blank_image==2)

# ret,img = cv2.threshold(img,127,255,0)
img=cl_img

size = np.size(img)
skel = np.zeros(img.shape,np.uint8)

element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
done = False
 
while( not done):
    eroded = cv2.erode(img,element)
    temp = cv2.dilate(eroded,element)
    temp = cv2.subtract(img,temp)
    skel = cv2.bitwise_or(skel,temp)
    img = eroded.copy()
 
    zeros = size - cv2.countNonZero(img)
    if zeros==size:
        done = True
plt.imshow(skel)

#%%
from skimage import data, filters, measure, morphology
szb=blank_image.shape
area_total=szb[0]*szb[1]



area_back=np.sum(np.uint8(blank_image==0))
area_spec=area_total-area_back
area_crack=np.sum(np.uint8(blank_image==2))
area_pore=np.sum(np.uint8(blank_image==3))
area_skel=np.sum(skel)

ratio_crack=area_crack/area_spec
ratio_pore=area_pore/area_spec

avg_thickness=area_crack/area_skel

# tmp_label=np.uint8(blank_image==2                   )
labels = measure.label(np.uint8(blank_image==3))


cl_img=np.uint8(blank_image==2)


#%%

from skimage.draw import ellipse
from skimage import data, filters, measure, morphology
from skimage.transform import rotate
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as io
io.renderers.default='browser'

fig = px.imshow(imgo, binary_string=True)
fig.update_traces(hoverinfo='skip') # hover is only for label info

# labels=blank_image
# labels=np.uint8(blank_image==3)
tmp_label=np.uint8(blank_image==2)
labels = measure.label(tmp_label)

props = measure.regionprops(labels, imgo)
properties = ['area', 'eccentricity', 'perimeter', 'intensity_mean']
for index in range(1, labels.max()):
    label_i = props[index].label
    contour = measure.find_contours(labels == label_i, 0.5)[0]
    y, x = contour.T
    hoverinfo = ''
    
    for prop_name in properties:
        fig.add_trace(go.Scatter(
        x=x, y=y, name=label_i,
        mode='lines', fill='toself', showlegend=False,
        hovertemplate=hoverinfo, hoveron='points+fills'))
    
plotly.io.show(fig)
