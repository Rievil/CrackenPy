"""
Created on Mon Dec 11 15:38:21 2023

@author: Richard
"""
import pandas as pd
from CrackManager import CrackManager as CrackManager
import cv2
import numpy as np

dfb=pd.read_excel(r'..\Classes.xlsx',sheet_name='Priorita')
#%

mask= dfb['Binder'].isnull()
dfb=dfb[~mask]
dfb=dfb.reset_index()
#%%
dfb['Folder'].unique()
#%%
for row in range(3,14):
    path=r'C:\PyTorchData\Camera\{:s}\Images'.format(dfb['Folder'][row])
    
    
    cm=CrackManager(path)
    
    cm.SetIdx(count=100)
    
    rec=[dfb['Dim_1'][row],dfb['Dim_2'][row],dfb['Dim_3'][row],dfb['Dim_4'][row]]
    rec=np.array(rec,dtype='int')
    ys=rec[0]
    ye=rec[2]+rec[0]
    xs=rec[1]
    xe=rec[3]+rec[1]
    crec=[xs,xe,ys,ye]
    
    cm.SetCropDim(crec)
    
    result=cm.Scan()
    
    dfr=pd.DataFrame(result)
    sheetname="{:s}_{:s}_{:d}".format(dfb['Folder'][row],dfb['Binder'][row],dfb['Age'][row])
    
    with pd.ExcelWriter('Img_Measure.xlsx',mode='a',engine="openpyxl") as writer:
        dfr.to_excel(writer, sheet_name=row)
    
    
    # cm.delete()
    #%%
# sheetname="{:s}_{:s}_{:d}_{:d}".format(dfb['Folder'][row],dfb['Binder'][row],dfb['Age'][row],dfb['Tensometer'][row])
with pd.ExcelWriter('Img_Measure.xlsx',mode='a',engine="openpyxl") as writer:
    dfb.to_excel(writer, sheet_name='Desc')
    
#%%
from matplotlib import pyplot as plt
from datetime import datetime, date, timedelta
df=pd.read_excel('Img_Measure.xlsx',sheet_name='Desc')

fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(4,6))
axes=[ax1,ax2,ax3]
for i,bid in enumerate(['C','H','WG']):

    dfb=df[df['Binder']==bid]
    
    for index, row in dfb.iterrows():
        if index==14:
            break
        
        dfn=pd.read_excel('Img_Measure.xlsx',sheet_name=str(index))
        x=dfn['date'].values
        # xi=x.
        time_epoch=x-x[0]
        
        # time = time_epoch.total_seconds() 
        
        time_epoch=time_epoch*1e-9
        # time_epoch=time_epoch+timedelta(day0s=1*index)
        time=np.array(time_epoch,dtype='f')
        time=time/(3600*24)
        # time_epoch
        # time_epoch=time_epoch
        
        y=dfn['crack_ratio']
        tit="{:d} {:d}".format(row['Age'],row['Tensometer'])
        cax=axes[i]
        cax.plot(time,y,label=tit)
        # cax.legend()
        cax.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
        cax.set_xlim(0,8)
        cax.set_ylim(0,0.038)
        cax.set_title(bid)
    

fig.supxlabel('Time [days]')
fig.supylabel('Crack ratio [-]')
plt.tight_layout()
plt.savefig(r'Plots\Binder_cack.pdf')
#%%
from matplotlib import pyplot as plt
from datetime import datetime, date, timedelta
df=pd.read_excel('Img_Measure.xlsx',sheet_name='Desc')

fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(4,6))
axes=[ax1,ax2,ax3]
for i,bid in enumerate(['C','H','WG']):

    dfb=df[df['Binder']==bid]
    
    for index, row in dfb.iterrows():
        if index==14:
            break
        
        dfn=pd.read_excel('Img_Measure.xlsx',sheet_name=str(index))
        x=dfn['date'].values
        # xi=x.
        time_epoch=x-x[0]
        
        # time = time_epoch.total_seconds() 
        
        time_epoch=time_epoch*1e-9
        # time_epoch=time_epoch+timedelta(day0s=1*index)
        time=np.array(time_epoch,dtype='f')
        time=time/(3600*24)
        # time_epoch
        # time_epoch=time_epoch
        
        y=dfn['crack_thickness']
        tit="{:d} {:d}".format(row['Age'],row['Tensometer'])
        cax=axes[i]
        cax.plot(time,y,label=tit)
        cax.legend()
        cax.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
        cax.set_xlim(0,8)
        cax.set_ylim(0.08,0.4)
        cax.set_title(bid)
    

fig.supxlabel('Time [days]')
fig.supylabel('Average crack thickness [mm]')
plt.tight_layout()
plt.savefig(r'Plots\Crack_Thickness.pdf')
#%%
from matplotlib import pyplot as plt
from datetime import datetime, date, timedelta
df=pd.read_excel('Img_Measure.xlsx',sheet_name='Desc')

fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(4,6))
axes=[ax1,ax2,ax3]
for i,bid in enumerate(['C','H','WG']):

    dfb=df[df['Binder']==bid]
    
    for index, row in dfb.iterrows():
        if index==14:
            break
        
        dfn=pd.read_excel('Img_Measure.xlsx',sheet_name=str(index))
        x=dfn['date'].values
        # xi=x.
        time_epoch=x-x[0]
        
        # time = time_epoch.total_seconds() 
        
        time_epoch=time_epoch*1e-9
        # time_epoch=time_epoch+timedelta(day0s=1*index)
        time=np.array(time_epoch,dtype='f')
        time=time/(3600*24)
        # time_epoch
        # time_epoch=time_epoch
        
        y=dfn['avg_pore_size']
        tit="{:d} {:d}".format(row['Age'],row['Tensometer'])
        cax=axes[i]
        cax.plot(time,y,label=tit)
        # cax.legend()
        cax.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
        cax.set_xlim(0,8)
        cax.set_ylim(0,600)
        cax.set_title(bid)
    

fig.supxlabel('Time [days]')
fig.supylabel('Average pore size [$mm^{2}$]')
plt.tight_layout()
plt.savefig(r'Plots\PoreSize.pdf')
#%%
from CrackPy import CrackPy
cp=CrackPy()
#%%
file=r'Img\ID14_144_Image.png'

mask=cp.GetImg(file)
cp.MeasureBW()

#%%
from matplotlib import pyplot as plt
fig,(ax1,ax2)=plt.subplots(2,1)
ax1.imshow(cp.img)

ax2.imshow(cp.img)
ax2.imshow(cp.mask,cmap='jet',alpha=0.4)


#%% Erode and dilate operations
import numpy as np
import cv2 
mask=cp.mask

mat_bw=mask[:,:]==1
mat_bw=mat_bw.astype(np.uint8)

kernel = np.ones((50, 50), np.uint8) 
mat_bw = cv2.dilate(mat_bw, kernel, iterations=1)
mat_bw = cv2.erode(mat_bw, kernel) 


pore_bw=cp.mask[:,:]==3
pore_bw=pore_bw.astype(np.uint8)



plt.imshow(cp.img)
# plt.imshow(pore_bw,alpha=0.8)


img_bwa = cv2.bitwise_and(mat_bw,pore_bw)

plt.imshow(img_bwa,alpha=0.4)



