#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import  train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


dataset=pd.read_csv("LinearRegression.csv",names=['X','Y'],skiprows=1)
dataset=pd.DataFrame(dataset)
#print(dataset)
#dataset.describe()

x=dataset['X'].values
x=x.reshape(-1,1)
y=dataset['Y'].values
y=y.reshape(-1,1)



# In[5]:



x_trainset,x_testset,y_trainset,y_testset=train_test_split(x,y,test_size=1/10, random_state=0)

print(len(x_trainset))
print(len(x_testset))

print(len(y_trainset))
print(len(y_testset))


len_trainset=len(x_trainset)
len_testset=len(x_testset)



# In[16]:



#dataset[i]
x_axis=list()
e_test=list()
e_train=list()

intervals=int(input("Enter the intervals:"))
intervals=int(intervals)
for i in range(int(intervals)):
    
    upto_train=((i+1)/intervals)*len_trainset
    upto_test=((i+1)/intervals)*len_testset
    if(int(upto_test)<=0): upto_test=int(1)
    if(int(upto_train)<=0): upto_train=int(1)
    
    print("Iteration "+str(i)+" Train: 0 TO "+ str(int(upto_train)))
    print("Iteration "+str(i)+" Test : 0 TO "+ str(int(upto_test)))
        
    x_train=x_trainset[0:int(upto_train)]
    x_test=x_testset[0:int(upto_test)]
    
    y_train=y_trainset[0:int(upto_train)]
    y_test=x_testset[0:int(upto_test)]
    
    
    regressor = LinearRegression()  
    regressor.fit(x_train,y_train)
    
    y_predict=regressor.predict(x_test)
    e_test.append(mean_squared_error(y_test,y_predict))
    y_predict_train=regressor.predict(x_train)
    
    e_train.append(mean_squared_error(y_train,y_predict_train))
    colors = (1,0,1)
    area = np.pi*3

    plt.scatter(x_train,y_train, s=area, c=colors)
    colors = (0,0,1)
    colors=np.array(colors)
    area = np.pi*3

    plt.scatter(x_test,y_test, s=area, c=colors)
    colors=np.array(colors)
    plt.plot(x_test,y_predict, color='green')
    plt.title('Scatter plot PREDICT  DATA DATASET'+str(i))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    x_axis.append(str(100*((i+1)/intervals))+"%")
    

print(e_train)
print(e_test)

plt.plot(x_axis, e_train)
plt.plot(x_axis, e_test)
plt.xlabel('Dataset No')
plt.ylabel('MSE')
plt.show()

print(x_axis)


# In[ ]:




