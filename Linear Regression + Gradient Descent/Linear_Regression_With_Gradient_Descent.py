
# coding: utf-8

# In[401]:


import numpy as np 
import matplotlib.pyplot as plt 
import numpy
import pandas as pd


# In[402]:


def findderivativeandupdate(X_train, Y_train,Theta,alpha):
    #print(Theta)
    #FINDIND DERIVATIVE
    
    sum1=0
    sum2=0    
    for i in range(len(X_train)):
        N=float(len(X_train))
        Predictedvalue= Theta[0] + Theta[1]*X_train
        Actualvalue= Y_train
        
        factor1= (-2/N) * sum(X_train*(Actualvalue-Predictedvalue))
        factor2= (-2/N) * sum(Actualvalue-Predictedvalue)     
        sum1=  factor1
        sum2=  factor2
        derivative_wrt_Theta_1= sum1
        derivative_wrt_Theta_0= sum2
        Theta0=float(Theta[0])-alpha*derivative_wrt_Theta_0
        Theta1=float(Theta[1])-alpha*derivative_wrt_Theta_1
    #print(Theta0,Theta1)
    return (Theta0,Theta1)


# In[403]:


def estimate_coef(x, y): 
    # number of observations/points 
    b_0=0
    b_1=0
    return(b_0, b_1) 


# In[404]:


def findtrainingerror(X_train,Y_train,Theta):
    #TRAINING ERROR
    sum1=0
    for i in range(len(X_train)): 
        predictedvalue= Theta[0] + Theta[1]*X_train[i]
        actualvalue= Y_train[i]        
        # COST FUNCTION (J)= (PREDICTED VALUE - ORIGINAL VALUE)^2        
        factor= (actualvalue-predictedvalue)
        sum1= sum1 + factor*factor
    costfunction=sum1/len(X_train)
    train_error= sum1/len(X_train)
    return (train_error)  


# In[405]:


def findtestingerror(X_test,Y_test,Theta):
#TESTING ERROR 
    # observations 
    sum1=0
    for i in range(len(X_test)):
        factor= (Y_test[i] - Theta[0] - Theta[1]*X_test[i])
        sum1= sum1 + factor*factor
    test_error= sum1/len(X_test)
    return (test_error)    


# In[406]:


def plot_regression_line(x, y, b): 
    # plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "m", 
            marker = "o", s = 30) 

    # predicted response vector 
    y_pred = b[0] + b[1]*x 

    # plotting the regression line 
    plt.plot(x, y_pred, color = "g") 

    # putting labels 
    plt.xlabel('x') 
    plt.ylabel('y') 

    # function to show plot 
    plt.show() 


# In[407]:


def linear_regression_grad_descent(alpha,iterations):
    
    data=pd.read_csv("train_data.csv",names=['X','Y'],skiprows=1)
    data=pd.DataFrame(data)
    
    test_data=pd.read_csv("test_data.csv",names=['X','Y'],skiprows=1)
    test_data=pd.DataFrame(test_data)
    
    
    X_train=data['X'].values
    Y_train=data['Y'].values

    X_test=test_data['X'].values
    Y_test=test_data['Y'].values
    

    #print(data)
    #print(x)
    #print(y)
    #for i in range(len(data)):
    #x[i]=np.array(data[i][0])
    #    y[i]=np.array(data[i][1])
    # estimating coefficients
    Theta=np.array([0,0])
    Theta= estimate_coef(X_train, Y_train)
    #print(Theta)    
    print("Initially Estimated coefficients:\nTheta_0 = {}\nTheta_1 = {}".format(Theta[0], Theta[1]))
      

    x_axis=list()    
    error_train=list()
    error_test=list() 
    for i in range(iterations):
        Theta=findderivativeandupdate(X_train, Y_train,Theta,alpha)
        plot_regression_line(X_train, Y_train, Theta)
        trainingerror=findtrainingerror(X_train,Y_train,Theta)
        testingerror= findtestingerror(X_test,Y_test,Theta)
        error_train.append(trainingerror)
        error_test.append(testingerror)
        print("\n\nEpoch:" + str(i))
        print("Theta0 :"+ str(Theta[0])+ " Theta1:"+ str(Theta[1])+ "  Training Error: "+ str(trainingerror))
        
        x_axis.append(i)
    #print(error)

    plt.plot(x_axis,error_train)
    plt.plot(x_axis,error_test)
    plt.xlabel('Epoch Number ->')
    plt.ylabel('MSE -> ')
    plt.show()


# In[ ]:





# In[ ]:





# In[408]:


def main():
    learningrate=0.0001
    epocs=20
    linear_regression_grad_descent(learningrate,epocs)
if __name__ == "__main__": 
    main() 

