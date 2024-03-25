# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 13:52:36 2024

@author: Richard
"""

from crackest import cracks as cr
 
#%s
cp=cr.CrackPy(model=1) #Model optimized also for pores
 
#%
 
imfile=r'Examples\Img\ID14_940_Image.png' #Read a file
cp.GetMask(imfile)
 
#%
pc=cr.CrackPlot(cp)
pc.overlay(figsize=(16,4))
pc.Save(r'Examples\Plots\Example.png')
