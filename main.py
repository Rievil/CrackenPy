# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:37:43 2023

@author: dvorr
"""


from crackest import cracks as cr


cp=cr.CrackPy(model=1)


#%%
imfile=r'Img\ID14_470_Image.png'



#%%

pc=cr.CrackPlot(cp)
#%%

fig=pc.overlay(figsize=(17,6))

#%%

pc.Save(fig,'Example')
#%%

pc.distancemap()



#%%

cp.GetMask(r'Img\ID14_470_Image.png')
#%%

cp.SetRatio(length=160,width=40)
#%% Ulozeni spravne formatovaneho modelu pro matlab
import torch.onnx
X = torch.rand(1,3,416,416)
traced_model = torch.jit.trace(cp.model,X)
traced_model.save("Models\TracedModel.pt")

#%%

import torch

print(torch.__version__)

#%% List all models (files) of crackpy
import gdown
import pkg_resources
import crackpy_models
import os

model=pkg_resources.resource_listdir('crackpy_models', '')
online_models={'0':['resnext101_32x8d_N387_C5_30102023.pt','1AtTrLmDf7kmlfEbGEJ5e43_aa0SnGntL'],
             '1':['resnext101_32x8d_N387_C5_310124.pt','1qmAv34aIPRLCRGEG3gwbbsYQTYmZpnp5']}

check=[]
count_d=0
for key in online_models:
    count=model.count(online_models[key][0])
    if count==0:
        count_d+=1
        ids = "0B9P1L--7Wd2vNm9zMTJWOGxobkU"
        module_path=crackpy_models.__file__
        tar_folder=os.path.dirname(module_path)
        
        out_file=r'{:s}\{:s}'.format(tar_folder,online_models[key][0])
        url_id=online_models[key][1]
        print("Downloading deep learing model '{:s}' for module crackpy".format(online_models[key][0].replace('.pt','')))
        gdown.download(id=url_id, output=out_file, quiet=False)

if count_d==0:        
    print("All models are already downloaded")
else:
    print("Downloaded {:d} models".format(count_d))
        
#%%

from cracks import cracks as cr

cr.UpdateModels()

# model_exist=pkg_resources.resource_exists('models', r'resnext101_32x8d_N387_C5_30102023.pt')    
# model_path=pkg_resources.resource_string('models', r'resnext101_32x8d_N387_C5_30102023.pt')

#%%
import os
import crackpy_models  # Replace 'your_module' with the actual module name

module_path = crackpy_models.__file__
print(f"Path of the module: {module_path}")

# To get the directory containing the module
module_directory = os.path.dirname(module_path)
print(f"Directory of the module: {module_directory}")
#%%

import pkg_resources

def list_package_resources(package_name):
    try:
        # Get the distribution object for the specified package
        distribution = pkg_resources.get_distribution(package_name)
        
        # Get the path to the package
        package_path = distribution.location
        
        # List all the resources in the package
        resources = pkg_resources.resource_listdir(package_name, '')
        
        # Print or process the list of resources
        for resource in resources:
            print(resource)

    except pkg_resources.DistributionNotFound:
        print(f"Package '{package_name}' not found.")

# Replace 'your_package_name' with the actual name of your package
list_package_resources('models')
#%%
from matplotlib import pyplot as plt
import cv2

path=r'Img'
file_img="{:s}\\{:s}".format(path,'ID17_1_Image.jpg')
file_label="{:s}\\{:s}".format(path,'ID17_1_Label.tif')


img = cv2.imread(file_img)

mask = cv2.imread(file_label, cv2.IMREAD_GRAYSCALE)
# mask = cv2.imread(file_label)

plt.imshow(img)
plt.imshow(mask,alpha=0.7,cmap='jet')

#%% Ilustration of segmentation of an sample image

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from matplotlib.colors import ListedColormap

# colors = ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8", "#253494"]
colors = ["#0027B9", "#0DC9E7", "#E8DD00","#D30101"]
my_cmap = ListedColormap(colors, name="my_cmap")

fig,ax=plt.subplots(1,1,figsize=(5,4))

ax = plt.gca()
ax.imshow(cp.img)

im=ax.imshow(cp.mask,alpha=0.7,cmap=my_cmap)


divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
   

ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
 
cbar=plt.colorbar(im, cax=cax)
cbar.set_ticks([0,1,2,3])
cbar.ax.set_yticklabels(["Back","Matrix","Crack","Pore"])
cbar.ax.tick_params(labelsize=10,size=0)

ax.axis("off")

fig.savefig('Plots\DeepLearning_example_mask.png',dpi=300,bbox_inches = 'tight',
    pad_inches = 0)

plt.show()
#%% Test accuracy

maskfile=r'Img\14_WG2_470_Mask_cropped.tiff'



import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image

# colors = ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8", "#253494"]
colors = ["#434343", "#28F240", "#F118A9","#4763F0"]
my_cmap = ListedColormap(colors, name="my_cmap")
gtim=Image.open(maskfile)

fig,(ax,ax1,ax2)=plt.subplots(1,3,figsize=(11,4))

ax.imshow(cp.img)
ax.set_title('Input image')
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
# ax.axis("off")

ax1.imshow(cp.img)
ax1.imshow(gtim,alpha=0.7,cmap='jet')
ax1.set_title('Ground truth')
ax1.get_xaxis().set_ticks([])
ax1.get_yaxis().set_ticks([])
# ax1.axis("off")


ax2.imshow(cp.img)

im=ax2.imshow(cp.mask,alpha=0.7,cmap='jet')


divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
   

ax2.get_xaxis().set_ticks([])
ax2.get_yaxis().set_ticks([])
 
cbar=plt.colorbar(im, cax=cax)
cbar.set_ticks([0,1,2,3])
cbar.ax.set_yticklabels(["Background","Matrix","Crack","Pore"])
# cbar.ax.tick_params(labelsize=10)

ax2.set_title('Segmentation result')


fig.savefig('Plots\Class_Test_Acuracy.pdf',dpi=300,bbox_inches = 'tight',
    pad_inches = 0)

plt.show()

#%% IOU of all calsess on sample img
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import cv2
classlist=["Back","Matrix","Cracks","Pores"]

fig,axes=plt.subplots(nrows=2, ncols=2,figsize=(10,7))

colors = ["#000000", "#35FF32", "#FF0E91","#0074FF"]
my_cmap = ListedColormap(colors, name="my_cmap")

mIOU=0
for class_num,ax in zip([0,1,2,3],axes.flat):
    ax.imshow(cp.img)
    
    # class_num=3
    
    mask=np.array(cp.mask)
    bw_mask=mask[:,:]==class_num
    
    gtim_np = np.array(gtim)
    bw_mask_gt=gtim_np[:,:]==class_num
    bw_mask_gt=bw_mask_gt.astype(np.uint8)
    
    
    
    
    mask_r=np.zeros(bw_mask.shape)
    cs_mask_sum=0
    for i in range(0,mask_r.shape[0]):
        for j in range(0,mask_r.shape[1]):
            value=0
            if (bw_mask[i,j]==1) & (bw_mask_gt[i,j]==1):
                value=1 #true true
                cs_mask_sum+=1
            elif (bw_mask[i,j]==1) & (bw_mask_gt[i,j]==0):
                value=2 #true false
            elif (bw_mask[i,j]==0) & (bw_mask_gt[i,j]==1):
                value =3 #false true
            else:
                value=0
          
            mask_r[i,j]=value
    
    
    seg_mask_sum=bw_mask.sum()
    gt_mask_sum=bw_mask_gt.sum()
    
    IOU=cs_mask_sum/(seg_mask_sum+gt_mask_sum-cs_mask_sum)
    mIOU=mIOU+IOU
    im=ax.imshow(mask_r,alpha=0.8,cmap=my_cmap,interpolation='antialiased')
    
    
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    
    ax.set_title(r"Class: '{:s}' IOU:{:0.2f}".format(classlist[class_num],IOU))

plt.suptitle("Mean intersection over union {:s}: {:0.4f}".format(r"$\overline{IOU}$",mIOU/4))

cbar=fig.colorbar(im, ax=axes.ravel().tolist())
# cbar=plt.colorbar(im, cax=cax)
cbar.set_ticks([0,1,2,3])
cbar.ax.set_yticklabels(["Empty","True Positive","False Positive","False Negative"])
fig.savefig('Plots\IOU_Example.pdf',dpi=300,bbox_inches = 'tight',
    pad_inches = 0)

plt.show()

#%% Example of ixel mm conversion
imfile=r'Img\ID14_940_Image.png'
mask=cp.GetImg(imfile)
img=cp.img
bw_props=cp.MeasureBW()

#%%
from skimage.measure import label, regionprops, regionprops_table
import pandas as pd
import math
import numpy as np
from matplotlib import pyplot as plt
# fig,ax=plt.subplots(1,1)

# ax.imshow(img)
# ax.imshow(mask,alpha=0.7,cmap='jet')
# ax.get_xaxis().set_ticks([])
# ax.get_yaxis().set_ticks([])

reg_props=('area','centroid','orientation','axis_major_length','axis_minor_length','bbox')

length=160
width=40


mask = np.array(mask)
bw_mask=mask[:,:]==0
bw_mask=~bw_mask

image=bw_mask.astype(np.uint8)
label_img = label(image)
# regions = regionprops(label_img)

props_mat = regionprops_table(label_img, properties=reg_props)
dfmat=pd.DataFrame(props_mat)
dfmat.sort_values(by=['area'],ascending=True)
dfmat=dfmat.reset_index()

#%%
fig, (ax,ax1) = plt.subplots(nrows=2,ncols=1)
ax.imshow(image, cmap=plt.cm.gray)
#
for index,props in dfmat.iterrows():
    # y0, x0 = props.centroid-0
    if props['area']>1000:
        y0=props['centroid-0']
        x0=props['centroid-1']
        
        
        orientation = props['orientation']
        
        rat1=0.43
        x0i = x0 - math.cos(orientation) * rat1 * props['axis_minor_length']
        y0i = y0 + math.sin(orientation) * rat1 * props['axis_minor_length']
        
    
        
        x1 = x0 + math.cos(orientation) * rat1 * props['axis_minor_length']
        y1 = y0 - math.sin(orientation) * rat1 * props['axis_minor_length']
        
        rat2=0.43
        x2i = x0 + math.sin(orientation) * rat2 * props['axis_major_length']
        y2i = y0 + math.cos(orientation) * rat2 * props['axis_major_length']
        
        x2 = x0 - math.sin(orientation) * rat2 * props['axis_major_length']
        y2 = y0 - math.cos(orientation) * rat2 * props['axis_major_length']
    
        ax.plot((x0i, x1), (y0i, y1), '-r', linewidth=2.5)
        ax.plot((x2i, x2), (y2i, y2), '-r', linewidth=2.5)
        ax.plot(x0, y0, '.g', markersize=15)
    
        minr = props['bbox-0']
        minc = props['bbox-1']
        maxr = props['bbox-2']
        maxc = props['bbox-3']
        
        
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, '-g', linewidth=2.5,alpha=0.7)
    # break

ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

ax1.imshow(cp.img)
ax1.imshow(cp.mask,alpha=1,cmap='jet')
ax1.get_xaxis().set_ticks([])
ax1.get_yaxis().set_ticks([])

fig.savefig('Plots\Pixel_mm_Ratio.pdf',dpi=300,bbox_inches = 'tight',
    pad_inches = 0)
# ax.axis((0, 600, 600, 0))
plt.show()
#%%
df=pd.DataFrame(bw_props,index=[0])
df=df.T
df.reset_index(inplace=True)

print(df.to_latex(index=False,
                  formatters={"name": str.upper},
                  float_format="{:.1f}".format,
)) 
#%%
from matplotlib import pyplot as plt

from skimage import data
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform
from skimage.draw import ellipse


# =============================================================================
# # Sheared checkerboard
# tform = AffineTransform(scale=(1.3, 1.1), rotation=1, shear=0.7,
#                         translation=(110, 30))
# image = warp(data.checkerboard()[:90, :90], tform.inverse,
#               output_shape=(200, 310))
# # Ellipse
# rr, cc = ellipse(160, 175, 10, 100)
# image[rr, cc] = 1
# # Two squares
# image[30:80, 200:250] = 1
# image[80:130, 250:300] = 1
# =============================================================================
image=bw_mask
coords = corner_peaks(corner_harris(image), min_distance=100, threshold_rel=0.02)
coords_subpix = corner_subpix(image, coords, window_size=13)



fig, ax = plt.subplots()
ax.imshow(image, cmap=plt.cm.gray)
ax.plot(coords[:, 1], coords[:, 0], color='cyan', marker='o',
        linestyle='None', markersize=6)
ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
# ax.axis((0, 310, 200, 0))
plt.show()
#%%
dfmat.sort_values(by=['area'],ascending=False)
dfmat=dfmat.reset_index()


l_rat=length/dfmat['axis_major_length'][0]
w_rat=width/dfmat['axis_minor_length'][0]
m_rat=(l_rat+w_rat)/2
pixel_mm_ratio=m_rat




#%%

from skimage.morphology import skeletonize
from skimage import data
import sknw

img = bw_mask
ske = skeletonize(bw_mask, method='lee')

# build graph from skeleton
graph = sknw.build_sknw(ske)

# draw image
plt.imshow(img, cmap='gray')

# draw edges by pts
for (s,e) in graph.edges():
    ps = graph[s][e]['pts']
    plt.plot(ps[:,1], ps[:,0], 'green')
    
# draw node by o
nodes = graph.nodes()
ps = np.array([nodes[i]['o'] for i in nodes])
plt.plot(ps[:,1], ps[:,0], 'r.')

plt.imshow(cp.img)        
plt.imshow(bw_mask,alpha=0.5,cmap='jet')
# title and show
plt.title('Build Graph')
plt.show()
#%%
# imfile=r'C:\Users\dvorr\VUT\PomVedi - General\Data\FIJI\14_WG_2\ID14_470_Image.png'
# imfile=r'C:\Users\dvorr\VUT\PomVedi - General\Data\FIJI\5_WG_8\ID5_168_Image.png'
# imfile=r'C:\Users\dvorr\VUT\PomVedi - General\Data\FIJI\3_WG_1\ID3_184_Image.png'
# imfile=r'Img\354_Image_04-10-2023 05-44-07.png'
# mask=cp.GetImg(imfile)

meas=cp.MeasureBW()
#%%

fig,ax=plt.subplots(1,1)
ax.imshow(cp.img)
ax.imshow(cp.mask,alpha=0.8,cmap='jet')

tit_str="Crack ratio: {:.4f}, Length: {:.0f}, Thickness: {:.2f}".format(meas["CrackRatio"],
                                                                   meas["TotalCrackLength"],
                                                                   meas["MeanCrackThickness"])
plt.title(tit_str)
print(tit_str)
#%%
mask=np.array(cp.mask)
bw_mask=mask[:,:]==2
bw_mask=bw_mask.astype(np.uint8)

import cv2
from matplotlib import pyplot as plt

def draw_image_histogram(image, channels, color='k'):
    hist = cv2.calcHist([image], channels, None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
    
def show_grayscale_histogram(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    draw_image_histogram(grayscale_image, [0])
    plt.show()

show_grayscale_histogram(cp.img)
    

#%%

import plotly
import plotly.express as px
import plotly.graph_objects as go
from skimage import data, filters, measure, morphology


img = cp.img
# Binary image, post-process the binary mask and compute labels
mask=np.array(cp.mask)
# bw_mask=mask[:,:]==2
# mask=bw_mask.astype(np.uint8)
labels = measure.label(mask)

fig = px.imshow(img, binary_string=True)
fig.update_traces(hoverinfo='skip') # hover is only for label info

props = measure.regionprops(labels, img)
properties = ['area', 'eccentricity', 'perimeter', 'intensity_mean']

# For each label, add a filled scatter trace for its contour,
# and display the properties of the label in the hover of this trace.
for index in range(1, labels.max()):
    label_i = props[index].label
    contour = measure.find_contours(labels == label_i, 0.5)[0]
    y, x = contour.T
    hoverinfo = ''
    for prop_name in properties:
        hoverinfo += f'<b>{prop_name}: {getattr(props[index], prop_name):.2f}</b><br>'
    fig.add_trace(go.Scatter(
        x=x, y=y, name=label_i,
        mode='lines', fill='toself', showlegend=False,
        hovertemplate=hoverinfo, hoveron='points+fills'))

plotly.io.show(fig)
#%%

from CrackManager import CrackManager as CrackManager

cm=CrackManager(r'C:\PyTorchData')
#%%
result=cm.Scan()
#%%
from skimage.measure import label, regionprops, regionprops_table
import math
import pandas as pd
from scipy.stats import skew, kurtosis
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist

# imfile=r'Img\ID14_940_Image.png'
mask=cp.GetImg(imfile)
cp.GetRatio()
cp.MeasureBW()




reg_props=('area','centroid','orientation','axis_major_length','axis_minor_length')

length=160
width=40

bw_mask=cp.masks['mat']
# mask = np.array(cp.mask)
# bw_mask=mask[:,:]==1
image=bw_mask.astype(np.uint8)
label_img = label(image)

props_mat = regionprops_table(label_img, properties=reg_props)
dfmat=pd.DataFrame(props_mat)
dfmat.sort_values(by=['area'],ascending=False)
dfmat=dfmat.reset_index()


l_rat=length/dfmat['axis_major_length'][0]
w_rat=width/dfmat['axis_minor_length'][0]
m_rat=(l_rat+w_rat)/2



label_img_pore = cp.masks['pore']
image_pore=label_img_pore

props_pore = regionprops_table(label_img_pore, properties=reg_props)
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

#%%

