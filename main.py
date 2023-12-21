# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:37:43 2023

@author: dvorr
"""

from CrackPy import CrackPy as CrackPy
import torch.onnx

imfile=r'Img\ID14_144_Image.png'

cp=CrackPy()

#%%

imfile=r'Img\name_104.png'
mask=cp.GetImg(imfile)

#%% Ulozeni spravne formatovaneho modelu pro matlab
X = torch.rand(1,3,416,416)
traced_model = torch.jit.trace(cp.model,X)
traced_model.save("Models\TracedModel.pt")

#%%

import torch

print(torch.__version__)

#%%

import matplotlib.pyplot as plt

#imfile=r'Img\021.jpg'
imfile=r'Img\name_104.png'
mask=cp.GetImg(imfile)


plt.imshow(cp.img)

plt.imshow(cp.mask,alpha=0.8,cmap='jet')

#%%

import matplotlib.pyplot as plt
import numpy as np

plt.imshow(cp.img)
mask=np.array(cp.mask)
bw_mask=mask[:,:]==2
bw_mask=bw_mask.astype(np.uint8)
plt.imshow(bw_mask,alpha=0.8,cmap='jet',interpolation='antialiased')
#%%
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.morphology import medial_axis, skeletonize

 
img = cp.img


mask=np.array(cp.mask)
bw_mask=mask[:,:]==2
bw_mask=bw_mask.astype(np.uint8)

skel_lee = skeletonize(bw_mask, method='lee')

plt.imshow(cp.img)        
plt.imshow(bw_mask,alpha=0.5,cmap='jet')
plt.imshow(skel_lee,alpha=0.5,cmap='jet',interpolation='antialiased')

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

imfile=r'Img\ID14_940_Image.png'
mask=cp.GetImg(imfile)
cp.GetRatio()





reg_props=('area','centroid','orientation','axis_major_length','axis_minor_length')

length=160
width=40

mask = np.array(cp.mask)
bw_mask=mask[:,:]==1
image=bw_mask.astype(np.uint8)
label_img = label(image)

props_mat = regionprops_table(label_img, properties=reg_props)
dfmat=pd.DataFrame(props_mat)
dfmat.sort_values(by=['area'],ascending=False)
dfmat=dfmat.reset_index()


l_rat=length/dfmat['axis_major_length'][0]
w_rat=width/dfmat['axis_minor_length'][0]
m_rat=(l_rat+w_rat)/2


bw_mask=mask[:,:]==3
image_pore=bw_mask.astype(np.uint8)
label_img_pore = label(image_pore)

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

dfa=pd.DataFrame({"dist":arr})

#%%

import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

points = np.array([dfpores['centroid-1'],dfpores['centroid-0']])
points=np.rot90(points)

simplices = Delaunay(points).simplices
plt.imshow(cp.img)
plt.imshow(image_pore,alpha=0.5)
plt.triplot(points[:, 0], points[:, 1], simplices)
plt.scatter(points[:, 0], points[:, 1], color='r')

arr=pdist(points,metric='minkowski')

plt.title("{:0.2f}".format(arr.mean()*cp.pixel_mm_ratio))
plt.show()


#%%
fig, ax = plt.subplots()
import seaborn as sns

sns.histplot(data=dfa, x="dist",
             element="bars", fill=True)

ar_skew=skew(dfpores['area'].values)
ar_kurt=kurtosis(dfpores['area'].values)
plt.title("Skew: {:0.3f} skew: {:0.3f}".format(ar_skew,ar_kurt))


