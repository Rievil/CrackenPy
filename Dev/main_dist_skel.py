# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 10:52:15 2024

@author: user
"""


from skimage.morphology import medial_axis, skeletonize
from matplotlib import pyplot as plt
from cracks import cracks as cr


cp=cr.CrackPy()
#%%

cp.GetMask(r'Img\14_WG2_470_Img_cropeed.png')
#%%
# cp.GetImg('C:\PyTorchData\Crack_Examples\Img\14_WG2_470_Img_cropeed.png')
from skimage.morphology import medial_axis, skeletonize
from matplotlib import pyplot as plt
import numpy as np

crack_bw=cp.mask[:,:]==2
crack_bw=crack_bw.astype(np.uint8)

skel = skeletonize(crack_bw, method='lee')

plt.imshow(cp.img)
plt.imshow(cp.mask,alpha=0.5,cmap='jet')
plt.imshow(skel,alpha=0.5)
#%%

import cv2
import numpy as np
import skimage.morphology

# read input
# img = cv2.imread('s_curve.jpg')

# convert to grayscale
img=cp.img
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# use thresholding
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

# get distance transform
distance = thresh.copy()
distance = cv2.distanceTransform(distance, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)

# get skeleton (medial axis)
binary = thresh.copy()
binary = binary.astype(np.float32)/255
skeleton = skimage.morphology.skeletonize(binary).astype(np.float32)

# apply skeleton to select center line of distance 
thickness = cv2.multiply(distance, skeleton)

# get average thickness for non-zero pixels
average = np.mean(thickness[skeleton!=0])

# thickness = 2*average
thick = 2 * average
print("thickness:", thick)
#%%

plt.imshow(cp.img)
plt.imshow(thickness,cmap='jet',alpha=0.7)

#%% Get mask for skeleton and for distance map


import cv2 
import numpy as np 


cp.GetImg('C:\Data\PyTorch\14_WG2_470_Img_cropeed.png')
#%%
cp.GetRatio(160,40)

#%% Get distance map and skeleton
# crack_bw=cp.mask[:,:]==2
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle

mask=cp.mask
crack_bw=mask[:,:]==2
crack_bw=crack_bw.astype(np.uint8)

thresh=crack_bw
#Determine the distance transform. 
skel = skeletonize(thresh, method='lee')
dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5) 
idx=skel==1
dist_skel=dist[idx]

  


fig,(ax,ax1)=plt.subplots(nrows=1,ncols=2)
# Display the distance transform 
idxy=[227,227+40]
idxx=[263,263+40]

ax.imshow(cp.img)
ax.imshow(dist,cmap='jet',alpha=0.7) 

ax.add_patch(Rectangle((idxy[0], idxx[0]), 40, 40,
             edgecolor = 'white',
             facecolor = 'none',
             fill=False,
             lw=2))

# ax.imshow(skel,alpha=0.5)

ax1.imshow(cp.img[idxx[0]:idxx[1],idxy[0]:idxy[1]])
c=ax1.imshow(dist[idxx[0]:idxx[1],idxy[0]:idxy[1]],cmap='jet',alpha=0.7) 


# ax1.imshow(skel[idxx[0]:idxx[1],idxy[0]:idxy[1]],alpha=0.5)
# cv2.waitKey(0) 

divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
   
cbar=plt.colorbar(c,cax=cax,label='Crack skeleton thickness [pixels]')

for axn in [ax,ax1]:
    axn.get_xaxis().set_ticks([])
    axn.get_yaxis().set_ticks([])

arr_dist=dist[skel==1]*cp.pixel_mm_ratio*2
#%%

plt.hist(arr_dist,bins=20)
#%% Pick skeleton==1 in distance map

arr_map=dist
arr_map[skel!=1]=0

# plt.imshow( crack_bw) 
gray = cv2.cvtColor(cp.img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray,cmap='binary')

arr_map = np.ma.masked_where(arr_map ==0, arr_map)

c=plt.imshow(arr_map,cmap='jet')
plt.colorbar(c)
plt.clim(arr_map.min(),arr_map.max())

#%%

#%%
from skimage.morphology import skeletonize
from skimage import data
import sknw
import pandas as pd

# open and skeletonize
img =cp.img
ske =skel

# build graph from skeleton
graph = sknw.build_sknw(ske)

# draw image
plt.imshow(img, cmap='gray')

# draw edges by pts
m_nodes=pd.DataFrame()
for (s,e) in graph.edges():
    ps = graph[s][e]['pts']
    if ps.shape[0]>20:
        idxx=[0,ps.shape[0]-1]
        idxy=[0,0]
        idxyi=[1,1]
        
        yps=ps[idxx,idxy]
        xps=ps[idxx,idxyi]
        m_nodesi=pd.DataFrame({'x':xps,'y':yps},index=[0,1])
        m_nodes=pd.concat([m_nodes,m_nodesi],ignore_index=True)
        
        plt.scatter(xps,yps,color='red')
        plt.plot(ps[:,1], ps[:,0], 'green')
    

m_node_freq=m_nodes[['x','y']].value_counts().reset_index(name='count')

mask=m_node_freq['count']>1

nodes_in=m_node_freq[mask]

plt.scatter(nodes_in['x'].values,nodes_in['y'].values,color='lime',s=120)
    
# draw node by o
# nodes = graph.nodes()
# ps = np.array([nodes[i]['o'] for i in nodes])
# plt.plot(ps[:,1], ps[:,0], 'ro')


plt.show()
#%%




