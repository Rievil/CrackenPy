# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 17:22:17 2023

@author: dvorr
"""

from IEFrame import IESolver


ies=IESolver()




#%

file_list=ies.list_dir(r'C:\Users\dvorr\VUT\22-02098S - General\Data\Malty-rezonance-signaly')

#%%
print(file_list.index[0])
#%%
# fold=file_list['folder']
subdf=file_list.iloc[0:50]



#%%
"https://medium.com/codex/reading-files-fast-with-multi-threading-in-python-ff079f40fe56"

ies.read_multi()
#%%
import pandas as pd
from matplotlib import pyplot as plt



sig=ies.ReadSignal(25)
x,y=ies.get_fft(sig)

#%
# x=sig['time']
# y=sig['signal']
plt.plot(x,y)

#%%

import re

filename=sig['file']

oldstr='12 drummers drumming, 11 pipers piping, 10 lords a-leaping'

parts=filename.split('\\')
p = re.compile(r'(\d+\w)',re.IGNORECASE)
file=parts[-1]
file=file.replace('.csv','')

desc=file.split('-')

#%%

import multiprocessing
print("Number of cpu : ", multiprocessing.cpu_count())

#%%

from multiprocessing import Queue

colors = ['red', 'green', 'blue', 'black']
cnt = 1
# instantiating a queue object
queue = Queue()
print('pushing items to queue:')
for color in colors:
    print('item no: ', cnt, ' ', color)
    queue.put(color)
    cnt += 1

print('\npopping items from queue:')
cnt = 0
while not queue.empty():
    print('item no: ', cnt, ' ', queue.get())
    cnt += 1
    
#%%





