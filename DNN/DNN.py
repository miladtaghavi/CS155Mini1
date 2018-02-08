# Import libraries
import numpy as np
import matplotlib.pyplot as plt

import sklearn
import sklearn.model_selection

import tensorflow as tf 
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import BatchNormalization, LeakyReLU

# Load final validation data set
FV_data = np.loadtxt('../data/test_data.txt',delimiter=' ',skiprows=1)
# Load input training data set
train_data = np.loadtxt('../data/training_data.txt',delimiter=' ',skiprows=1)
utrain_data = np.unique(train_data,axis=0)

# Get header words list
f = open('../data/test_data.txt','r')
words = np.array(f.readline().split())
f.close()

# Splot y_train and x_train from training set
x_tall = train_data[:,1:]
y_tall = train_data[:,0]

x_uall = utrain_data[:,1:]
y_uall = utrain_data[:,0]

# One hot encode categories
y_tall = keras.utils.np_utils.to_categorical(y_tall)
y_uall = keras.utils.np_utils.to_categorical(y_uall)

# Function to generate DNN of given depth and width
def getModel(layers,Pdrop):
    model = Sequential()
    model.add(Dense(layers[0],input_shape=(1000,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(Pdrop))
    for i in layers[1:]:
        model.add(Dense(i))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
    
    # predicting probabilities of each of the 2 classes
    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model

# undo one hot encoding
def Unencode(out):
    ypred = out[:,0] < out[:,1]
    return ypred

# Function to get explicit model accuracy from softmax
def getAccuracy(model,xt,yt):
    out = model.predict(xt)
    ypred = Unencode(out)
    ytrue = Unencode(yt)
    acc = 1.0*np.sum(ypred == ytrue)/len(ytrue)
    return acc

# Function to get bag of words which were misclassified
def getBagOfWords(xtrain,ypred,ytrue,words):
    out = []
    ypredu = Unencode(ypred).astype(int)
    ytrueu = Unencode(ytrue).astype(int)
    # Get locations of bag of words which were misclassified
    idx = np.arange(0,len(ypredu))
    idxErr = idx[ypredu!=ytrueu]
    Xerr = xtrain[ypredu!=ytrueu]
    j = 0
    for i in Xerr:
        out.append([ytrue[idxErr[j]],ypred[idxErr[j]],words[i>0],i[i>0]])
        j=j+1
    return out

# Function to write final predictions
def writeResults(ypred):
    f = open('DNN_submission.txt','w')
    f.write('Id,Prediction\n')
    for i in range(0,len(ypred)):
        f.write(str(i+1)+','+str(int(ypred[i]))+'\n')
    f.close()

    # Specify number of Neural networks to train
N_models = 150
Predictions = []

# Define the DNN model
for i in range(0,N_models):
    print('Training DNN ',i)
    model = getModel([500,250,125],0.4)
    # Compile it and fit
    model.compile(loss='categorical_crossentropy',optimizer='RMSprop', metrics=['accuracy'])
    model.fit(x_uall, y_uall, batch_size=2**8, epochs=2)
    # Use weakly trained model to predict and store predictions
    ypred = model.predict(FV_data,batch_size=2**8)
    Predictions.append(ypred)


Predictions = np.array(Predictions)

ypred = []
for i in Predictions:
    ypred.append(Unencode(i).astype(int))

# Get mean and standard deviation of samples
ypmean=np.mean(ypred,axis=0)
std=np.std(ypred,axis=0)
print('Number of samples where stdev > 0, ',np.sum(std>0))

# Compute final predictions and output it
ypred = (ypmean > 0.5).astype(int)
writeResults(ypred)
