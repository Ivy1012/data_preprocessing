#!/usr/bin/env python
# coding: utf-8

# # data_preprocessing
# s2. nino3.4 index  
# extract the DJF month from the original data

# In[52]:


import numpy as np
import csv

def read_csv_2D(fpath,encoding = 'utf-8-sig'):
    # empty lists to hold the data
    date = []
    index = []
    # open the CSV file
    with open(fpath, 'r',encoding=encoding) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
    #------ save the nino3.4 (a float) from the second, third and the last column
            index.append(float(row[1]))
            index.append(float(row[2]))
            index.append(float(row[-1]))

            date.append(int(row[0])*100+1)
            date.append(int(row[0])*100+2)
            date.append(int(row[0])*100+12)
    return index,date

nino34_index,time = read_csv_2D('./data/nino3.4_index.csv')
# print(nino34_index)
date_index = np.zeros((len(nino34_index),2))
date_index[:,0] = time
date_index[:,1] = nino34_index

np.savetxt('./data/nino34_DJF_1870_2019.txt',date_index,fmt = '%6d %.2f',delimiter = ' ')

