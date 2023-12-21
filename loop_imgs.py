# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:35:54 2023

@author: Richard
"""

from CrackPy import CrackPy as CrackPy



cp=CrackPy()
#%%


# file_name='{:s}\{:s}'.format(dff['folder'][idx[i]],dff['name'][idx[i]])
cp.GetImg(r'C:\PyTorchData\ID2_184_Image.png')
# cp.GetImg(r'Plots\ID14_940_Image.png')
#%%
import matplotlib.pyplot as plt
fig,(ax1,ax)=plt.subplots(2,1,figsize=(7,5))
ax1.imshow(cp.img)
ax.imshow(cp.img)
ax1.axis("off")
c=ax.imshow(cp.mask,alpha=0.8,cmap='jet')
plt.axis("off")
cbar=fig.colorbar(c, ax=ax,ticks=[0, 1, 2, 3])
cbar.ax.set_yticklabels(['Back', 'Matrix', 'Cracks','Pores']) 
plt.tight_layout()
plt.savefig(r'Plots\SegmentExample.png',dpi=300)

#%%

m=cp.MeasureBW()
#%%
path=r'C:\PyTorchData\Camera\24052022_H2_1den'
import pandas as pd
import os
import datetime
from scipy import io
import numpy as np

mat_file=path +'\\CropDim.mat'
mat = io.loadmat(mat_file,simplify_cells=True)
dim=np.array(mat['dim'])
dim=dim.astype(int)
cp=CrackPy()
# cp.SetCropDim(dim)

im_path=path+'\\Images'

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
#%
import matplotlib.pyplot as plt
import cv2
from wand.image import Image as WI
from scipy import io
from tqdm.notebook import tqdm
from tqdm import trange



idxf=np.linspace(0,dff.shape[0],100)
idx = idxf.astype(int)



dfr=pd.DataFrame()
for i in trange (0,idx.size-1):
    file_name='{:s}\{:s}'.format(dff['folder'][idx[i]],dff['name'][idx[i]])
    mask=cp.GetImg(file_name)
    dc=cp.MeasureBW()
    dc['date']=dff['date'][idx[i]]
    dfn=pd.DataFrame(dc,index=[i])
    dfr=pd.concat([dfr,dfn],axis=0)
    
#%%
dfr.to_excel(r'Img_Measure.xlsx',sheet_name='09022022_H_WG_28dnu')

#%%

with pd.ExcelWriter('Img_Measure.xlsx',mode='a',engine="openpyxl") as writer:
    dfr.to_excel(writer, sheet_name='24052022_H2_1den')
    
#%%
import pandas as pd
from matplotlib import pyplot as plt
import datetime
from datetime import datetime
import numpy as np
from scipy.signal import savgol_filter

m_file='Img_Measure.xlsx'
xl = pd.ExcelFile(m_file)
fig,ax=plt.subplots(1,1,figsize=(4,4))
plt.grid(alpha=0.5)
names=xl.sheet_names
for name in names: 
    df=pd.read_excel(m_file,sheet_name=name)
    time=df['date'].values
    # time_oj= datetime.strptime(datetime_str, 
    #                              "%d%b%Y%H%M%S")
    time_epoch=time-time[0]
    time_epoch=time_epoch*1e-9/3600
    # delta=time[-1]-time[0]
    # print(delta)
    # print(delta)
    y=df['CrackRatio']
    yhat = savgol_filter(y, 5,2)
    # plt.plot(time_epoch,y,label=name)
    ax.plot(time_epoch,yhat,label=name)

plt.xlabel('Time [hours]')
plt.ylabel('Crack area ratio [-]')
plt.legend()
plt.tight_layout()    
plt.savefig(r'Plots\CrackRatio.pdf')
#%%
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,4))

x=dfr['date']
y=dfr['CrackArea']
y2=dfr['MeanCrackThickness']
y3=dfr['SpecArea']+dfr['PoreArea']
ax1.plot(x,y)

ax2.plot(x,y2)

#%%

plt.scatter(dfr['TotalCrackLength'],dfr['MeanCrackThickness'])

# plt.imshow(cp.img)
# plt.imshow(cp.mask,alpha=0.8,cmap='jet')

#%%
# file=r'C:\Users\Richard\Vysoké učení technické v Brně\22-02098S - Dokumenty\General\Data\Kamera\Measurements\09022022_H_WG_28dnu\Images\1_Image_12-02-2022 05-11-43.png'
# print(file_name)
from scipy import io
path=r'C:\Users\Richard\Vysoké učení technické v Brně\22-02098S - Dokumenty\General\Data\Kamera\Measurements\09022022_H_WG_28dnu'
mat_file=path +'\\CropDim.mat'
mat = io.loadmat(mat_file,simplify_cells=True)
dim=np.array(mat['dim'])
dim=dim.astype(int)


img = cv2.imread(r'Img\1_Image_12-02-2022 05-11-43.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img=img[dim[1]:dim[3],dim[0]:dim[2]]
plt.imshow(img)

#%%
i=50
file_name='{:s}\{:s}'.format(dff['folder'][idx[i]],dff['name'][idx[i]])
cp.GetImg(file_name)
# cp.
plt.imshow(cp.img)
plt.imshow(cp.mask,alpha=0.8,cmap='jet')
