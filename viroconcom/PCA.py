#!/usr/bin/env python
# coding: utf-8

# In[82]:


import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from imports import NDBCImport
import pandas as pd
#from IPython import display



class ViroconcomPCA:
    '''
    Takes csv and does PrincipalComponentAnalysis on data
    csv-form:  x,y,z
               int,int,int
    can do n-dimensional           
    
    provides method to convert csv with string into one with int()
    '''
    def _init_(self):
        
        #form prüfen
        
        pass
  


# In[83]:


# uses imports to do pca on NDBC data
# buoy = number of buoy in NDBC(usually 6 digit int)
# year = year of data(int)
# 
# only year
#
def pca_imports_year(self, buoy, year):
    buoy = NDBCImport(buoy)        
    buoy.get_virocon_data(year).to_csv("test.csv", index=False)        
    return buoy


# uses imports to do pca on NDBC data
# buoy = number of buoy in NDBC(usually 6 digit int)
# years = year range of data(int,int)
# 
# only range
#
def pca_imports_range(self, buoy, years):
    buoy = NDBCImport(buoy)
    buoy.get_virocon_data_range(years).to_csv("test.csv", index=False)
    return buoy
    
test = pca_imports_year(None, 41002, 2015)
test = open("test.csv", "r")
for line in test:
    print(line)
test.close()


# In[84]:


def extract_data_from_file(self):    
    data = open("test.csv", "r")
    data_array = []
    for line in data:
        #print(line)   
        array = []
        array = line.strip().split(",")     

    
        data_array.append(array)
 
    del data_array[0] 
    line_counter = 0
    for line in data_array:
        counter = 0
        for i in line:
            i = float(i)
            line[counter] = i
            counter = counter + 1
        data_array[line_counter] = line
        line_counter = line_counter + 1
    
    data.close()
    #print(data_array)
    return data_array
    
    

extract_data_from_file(None)


# In[105]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#import mglearn

def pca_on_data():
    
    data = extract_data_from_file(None)
    scaler = StandardScaler()
    scaler.fit(data)
    X_scaled = scaler.transform(data)
    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)
    
    print(X_pca)
    np.savetxt("test2.csv", X_pca, delimiter=",")
    #file = open("test2.csv", "w")
    #for line in X_pca:
    #    file.write(line + "\n")
    #file.close()
    
    
    
    
    
    print("Ursprüngliche Abmessung: {}".format(str(X_scaled.shape)))
    print("Reduzierte Abmessung: {}".format(str(X_pca.shape)))


    plt.figure(figsize=(8, 8))
    #mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1])
    plt.plot(X_pca[:, 0], X_pca[:, 1], 'go')
    #plt.legend(data, loc="best")
    plt.gca().set_aspect("equal")
    plt.xlabel("Erste Hauptkomponente")
    plt.ylabel("Zweite Hauptkomponente")
    
    
pca_on_data()


# In[99]:


from sklearn.datasets import load_iris
iris = load_iris()


scaler = StandardScaler()
scaler.fit(iris.data)
X_scaled = scaler.transform(iris.data)
pca = PCA(n_components=2)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

print("Ursprüngliche Abmessung: {}".format(str(X_scaled.shape)))
print("Reduzierte Abmessung: {}".format(str(X_pca.shape)))

plt.figure(figsize=(12, 12))
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], iris.target)

plt.legend(iris.target_names, loc="best")
plt.gca().set_aspect("equal")

plt.xlabel("Erste Hauptkomponente")
plt.ylabel("Zweite Hauptkomponente")


# In[ ]:





# In[ ]:




