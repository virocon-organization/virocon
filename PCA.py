#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler



class ViroconcomPCA:
    
    def __init__(self):
        pass
    
    def pca_on_data(self, data, numberOfDimension):        
        scaler = StandardScaler()
        scaler.fit(data)
        X_scaled = scaler.transform(data)
        pca = PCA(n_components=numberOfDimension)
        pca.fit(X_scaled)
        X_pca = pca.transform(X_scaled)
        return X_pca
    
    def get_data_array(self, data, dimensions):        
        x = self.pca_on_data(data, dimension)
        data_array = []
        n = dimensions-1
        
        for a in range(0, n):
            array = []
            for i in x:
                array.append(i[a])
            data_array.append(array) 
        return data_array      
        
        
        

