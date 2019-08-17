#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 10:07:13 2019

@author: ahmetkaanipekoren
"""


def data_preparation():
    """
    Reading the input fike
    """
    df = pd.read_csv("heart.csv")

    return df


def normalization(df):
    """
    Normalization
    """
    
    df_norm = (df-df.min()) / (df.max() - df.min())
    
    return df_norm


def x_y_values(df):
    df_y = df.loc[:,"target"]
    df_x = df.drop(["target"],axis=1)
    
    
    return df_x,df_y

    
def train_test(df_x,df_y):
    
    train_x,test_x,train_y,test_y = train_test_split(df_x,df_y,test_size = 0.2,random_state=42)
    train_x=train_x.T
    test_x = test_x.T
    
    train_x,test_x,train_y,test_y = np.array(train_x),np.array(test_x),np.array(train_y),np.array(test_y)
    train_y = train_y.reshape(1,242)
    test_y=test_y.reshape(1,61)
    return train_x,test_x,train_y,test_y




def initialize_parameters_NN(x):
    
    """
    Initialize parameters for 2 layer neural network, hidden layer with 3 nodes.
    """
    
    
    weight1 = np.random.randn(3,x.shape[0])
    bias1 = np.zeros((3,1))
    weight2 = np.random.randn(1,3)
    bias2 = np.zeros((1,1))
    
    parameters = {"weight1" : weight1,
                  "bias1" : bias1,
                  "weight2": weight2,
                  "bias2" : bias2}
    
    
    return parameters


def relu(z):
   
    return np.maximum(0, z)



def softmax(z):

    t = np.exp(z)
    return np.divide(t, np.sum(t, axis=0))



def sigmoid(z):
    
    return 1 / (1 + np.exp(-z))





def forward_propagation_NN(parameters,x):
    
    """
    Forward propagation calculations
    """
    
    Z1 = np.dot(parameters["weight1"],x) + parameters["bias1"]
    A1 = np.tanh(Z1)
    Z2 = np.dot(parameters["weight2"],A1) + parameters["bias2"]
    A2 = sigmoid(Z2)
    
    cache = {"Z1" : Z1,
             "A1" : A1,
             "Z2" : Z2,
             "A2" : A2}
    
    
    return A2,cache



def backward_propagation_NN(parameters, cache, X, Y):
    
    """
    Backward propagation calculations
    """

    dZ2 = cache["A2"]-Y
    dW2 = np.dot(dZ2,cache["A1"].T)/X.shape[1]

    db2 = np.sum(dZ2,axis =1,keepdims=True)/X.shape[1]
    dZ1 = np.dot(parameters["weight2"].T,dZ2)*(1 - np.power(cache["A1"], 2))
    dW1 = np.dot(dZ1,X.T)/X.shape[1]
    db1 = np.sum(dZ1,axis =1,keepdims=True)/X.shape[1]
    grads = {"dweight1": dW1,
             "dbias1": db1,
             "dweight2": dW2,
             "dbias2": db2}
    return grads


def update_parameters(grads,parameters,lr=0.01):
    
    """
    New parameters with learning rate
    """
    
    parameters ={
            "weight1" : parameters["weight1"] - lr * grads["dweight1"],
               "bias1" : parameters["bias1"] - lr * grads["dbias1"],
               "weight2" : parameters["weight2"] - lr * grads["dweight2"],
               "bias2" : parameters["bias2"] - lr* grads["dbias2"]
               
               }

    return parameters
    
    

def test(test_x,test_y):
    A2, cache = forward_propagation_NN(parameters, test_x)
    correct_prediction = 0
    check_list = []
    
    for i in range(len(A2[0])):
        if A2[0][i]>0.5:
            check_list.append(1)
        else:
            check_list.append(0)
            
    for i in range(len(A2[0])):
        if check_list[i] == test_y[0][i]:
            correct_prediction += 1
            
        else:
            pass
        
    acc = float(correct_prediction) / len(check_list)
    
    
    return acc,correct_prediction
        
    


def compute_cost_NN(A2, Y):
    cost_list = []
    for i in range(len(A2)):
        cost_list.append(0.5*(np.sqrt((A2[i]-Y[i])**2)))

    total = 0
    count = 0
    for num in cost_list:
        total = total + num
        count += 1

    return total / count






if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    df = data_preparation()
    df_norm = normalization(df)
    df_x,df_y = x_y_values(df_norm)
    train_x,test_x,train_y,test_y = train_test(df_x,df_y)
    iteration_list = []
    
    
    parameters = initialize_parameters_NN(train_x)
    cost_list = []
    for i in range(1000):
        A2,cache = forward_propagation_NN(parameters,train_x)
        grads = backward_propagation_NN(parameters, cache, train_x, train_y)
        parameters = update_parameters(grads,parameters,lr=0.1)
        cost = compute_cost_NN(A2,train_y)
        cost_list.append(np.mean(cost))
        iteration_list.append(i)
        
        
    acc,correct_prediction = test(test_x,test_y)
    
    acc_train, train_correct_prediction = test(train_x,train_y)
    
    
    import matplotlib.pyplot as plt
    plt.plot(iteration_list, cost_list,"ko")
    plt.show()
    
    
        
        
        
