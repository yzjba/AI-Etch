# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 10:52:27 2023

@author: yzj
"""
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
#from torchsummary import summary
import numpy as np
import pickle
from data_util import complot
import cv2
import time

from models import DNN,RNN,CRNN,CCNN,CNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def tvsplit(tn,sn,cn):
    trnidx = []
    validx = []
    for i in range(tn):
        if np.floor(i/cn)%sn==sn-1:
            validx.append(i)
        else:
            trnidx.append(i)
        
    return trnidx,validx

# Select the NN architecture
#net = DNN().to(device)
net = CRNN().to(device)
#net = CCNN().to(device)
#net = CNN().to(device)
print(net)

# load data
pkl_file = open('featbin.pkl', 'rb')
feat = pickle.load(pkl_file)
pkl_file.close()

print(np.min(feat,0),np.max(feat,0),np.mean(feat,0),np.std(feat,0))
print(np.mean(feat[:,4:]),np.std(feat[:,4:]))

# normalized by max value
mxv = np.max(feat,0)
mxv[4:]=np.max(mxv)
maxv = np.max(mxv)
feat = feat/np.tile(mxv,(10000,1))

X = np.zeros((200,48,366))
y = np.zeros((200,48,181))

for i in range(200):
    # ef is diff, ey is next step
    efeat = feat[i*50:(i+1)*50,:]
    ex = efeat[1:49,:]
    ey = efeat[2:50,4:]-efeat[1:49,4:]
    ey = np.maximum(ey,0)
    ef = efeat[1:49,4:]-efeat[0:48,4:]
    ef = np.maximum(ef,0)
    ex = np.concatenate([ex,ef],axis=1)
    X[i,:,:] = ex
    y[i,:,:] = ey

maxvd = 30
print('haha',np.mean(ey),np.min(ey),np.max(ey),np.std(ey))
print('haha',ex.shape,ey.shape)

indtt = np.array(range(3,200,4))
indtn = np.setdiff1d(np.array(range(0,200)),indtt)

#myv = np.max(y)
#y = y/myv

## input for DNN
#xtrain = X[indtn,:,:].reshape(-1,368,1,1)
#xtest = X[indtt,:,:].reshape(-1,368,1,1)

# input for RNN
fdim = 181
xtrain = np.zeros((len(indtn)*48,48,fdim))
xdtrain = np.zeros((len(indtn)*48,48,fdim))
xftrain = X[indtn,:,:4].reshape(-1,4)
for i in range(len(indtn)):
    for j in range(48):
        xtrain[i*48+j,47-j:48,:] = X[indtn[i],0:j+1,4:4+fdim]
        xdtrain[i*48+j,47-j:48,:] = X[indtn[i],0:j+1,4+fdim:4+2*fdim]
xtest = np.zeros((len(indtt)*48,48,fdim))
xdtest = np.zeros((len(indtt)*48,48,fdim))
xftest = X[indtt,:,:4].reshape(-1,4)
for i in range(len(indtt)):
    for j in range(48):
        xtest[i*48+j,47-j:48,:] = X[indtt[i],0:j+1,4:4+fdim]
        xdtest[i*48+j,47-j:48,:] = X[indtt[i],0:j+1,4+fdim:4+2*fdim]
xftrain,xftest = torch.FloatTensor(xftrain),torch.FloatTensor(xftest)

print('xshape:',xtrain.shape,xtest.shape, xftrain.shape, xftest.shape)

ytrain = y[indtn,:,:].reshape(-1,181)*maxvd
ytest = y[indtt,:,:].reshape(-1,181)*maxvd

xtrain,xtest,xdtrain,xdtest,ytrain,ytest = torch.FloatTensor(xtrain),torch.FloatTensor(xtest),torch.FloatTensor(xdtrain),torch.FloatTensor(xdtest),torch.FloatTensor(ytrain),torch.FloatTensor(ytest)

#print(summary(net,[xtrain,xftrain]))

#trainset = TensorDataset(xtrain,ytrain)
#testset = TensorDataset(xtest,ytest)

trainset = TensorDataset(xtrain,xdtrain,xftrain,ytrain)
testset = TensorDataset(xtest,xdtest,xftest,ytest)

[trnidx,validx] = tvsplit(7200,5,48)
validset = Subset(trainset, validx)#Subset类型
trainset = Subset(trainset, trnidx)#Subset类型

train_loader = DataLoader(trainset,batch_size=64,shuffle=True)
valid_loader = DataLoader(validset,batch_size=64,shuffle=True)
test_loader = DataLoader(testset,batch_size=48,shuffle=False)

# Function to test what classes performed well
def test_net():
    
    running_loss = 0.0
    
    with torch.no_grad():
        for i, (x, xd, xf, py) in enumerate(test_loader):
            py = py.to(device)
            x = x.to(device)
            xd = xd.to(device)
    #        output_y = net(x)
            xf = xf.to(device)
            output_y = net(x,xd,xf)
    
            loss = criterion(output_y, py)
            
            mse = torch.mean(torch.abs(output_y-py))
    
            
            # print loss statistics
            running_loss += mse.item()
            
            # 把base，预测值和实际值画在一个图上
#            print(x.shape)
            xbase = np.array(x[:,-1,:181].cpu()).reshape(181)*maxv
    #        xbase = np.array(x[:,6:187,:,:]).reshape(181)*maxv
    #        polar2im(xbase+np.array(output_y.detach()).reshape(181)*maxv,'predicted')
    #        polar2im(xbase+np.array(py).reshape(181)*maxv,'truth')
            
    #        compim(xbase,xbase+np.array(output_y.detach()).reshape(181)*maxv,xbase+np.array(py).reshape(181)*maxv)
#            if i%5000==19:
#                complot(xbase,xbase+np.array(output_y.detach().cpu()).reshape(181)*maxv/maxvd,xbase+np.array(py.cpu()).reshape(181)*maxv/maxvd)
    
    #        plt.figure()
    #        plt.plot(np.array(output_y.detach()).reshape(181),label='predicted')
    #        plt.plot(np.array(py).reshape(181),label='truth')
    #        plt.title('Etching')
    #        plt.legend(loc='upper right')
    #        plt.show()
        
    test_loss = running_loss/2400*maxv/maxvd
    return test_loss

def test_net_multi(numofstep):
    
    running_loss = np.zeros(numofstep)
    running_closs = np.zeros(numofstep)
    aerr = np.zeros((50,10,38,181))
    
    with torch.no_grad():
        for i, (x, xd, xf, py) in enumerate(test_loader):
            print(i)
#            for step in range(ns):
            for j in range (numofstep):
                
                if j>0: # get output by last prediction
#                    print(i,j,xt.shape,output_y.shape,y.shape,xdt.shape,xft.shape)
                    xt[:,:47,:] = xt[:,1:48,:]
                    xdt[:,:47,:] = xdt[:,1:48,:]
                    xt [:,47,:] = xt [:,46,:]+output_y/maxvd
                    xdt [:,47,:] = output_y/maxvd
                else: # zero
                    xt = x[j:j+48-numofstep,:,:]
                    xdt = xd[j:j+48-numofstep,:,:]
                    xft = xf[j:j+48-numofstep,:]
                
                y = py[j:j+48-numofstep,:]
                
                y = y.to(device)
                xt = xt.to(device)
        #        output_y = net(x)
                xft = xft.to(device)
                xdt = xdt.to(device)
        
                output_y = net(xt,xdt,xft)
                #print(i,j,xt.shape,output_y.shape)
        
                loss = criterion(output_y, y)
                
                mse = torch.mean(torch.abs(output_y-y))
#                print('mse',mse.shape,output_y.shape,y.shape)
                aerr[i,j,:,:] = np.array(torch.abs(output_y-y).cpu())
                
                xbaset = x[j:j+48-numofstep,-1,:181].to(device)
                xbasen = xt[:,-1,:181].to(device)
                cmse = torch.mean(torch.abs(xbasen+output_y/maxvd-xbaset-y/maxvd))
        
                # print loss statistics
                running_loss[j] += mse.item()
                running_closs[j] += cmse.item()
                
                # 把base，预测值和实际值画在一个图上
                #print(x.shape)
                xbase = np.array(xt[-1,47-j,:181].cpu()).reshape(181)*maxv
                xbasen = np.array(xt[-1,-1,:181].cpu()).reshape(181)*maxv
                xbaset = np.array(x[-1,j+47-numofstep,:181].cpu()).reshape(181)*maxv
        #        xbase = np.array(x[:,6:187,:,:]).reshape(181)*maxv
        #        polar2im(xbase+np.array(output_y.detach()).reshape(181)*maxv,'predicted')
        #        polar2im(xbase+np.array(py).reshape(181)*maxv,'truth')
                
        #        compim(xbase,xbase+np.array(output_y.detach()).reshape(181)*maxv,xbase+np.array(py).reshape(181)*maxv)
                if i%50==48:
                    complot(xbase,xbasen+np.array(output_y[-1,:181].detach().cpu()).reshape(181)*maxv/maxvd,xbaset+np.array(y[-1,:].cpu()).reshape(181)*maxv/maxvd)
    
    #        plt.figure()
    #        plt.plot(np.array(output_y.detach()).reshape(181),label='predicted')
    #        plt.plot(np.array(py).reshape(181),label='truth')
    #        plt.title('Etching')
    #        plt.legend(loc='upper right')
    #        plt.show()
        
    test_loss = running_loss/50*maxv/maxvd
    test_closs = running_closs/50*maxv
    aerr = aerr*maxv/maxvd
    return test_loss,test_closs,aerr


criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(net.parameters(), lr = 0.0003)

# Function to save the model
def saveModel(iepoch):
    path = "./processModelcuda"+str(iepoch)+".pth"
    torch.save(net.state_dict(), path)

def train_net(n_epochs):

    # prepare the net for training
    net.train()
    training_loss = []
    valid_loss = []
    test_loss = []

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        
       
        if epoch % 50 == 49:    # print every 10 batches
            saveModel(epoch)
            print('Epoch: {}, Avg. Loss: {}'.format(epoch + 1, running_loss))
        
        running_loss = 0.0
        # train on batches of data, assumes you already have train_loader
        for i, (x, xd, xf, py) in enumerate(train_loader):
            py = py.to(device)
            x = x.to(device)
            xf = xf.to(device)
            xd = xd.to(device)
    
            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            
            # print(images.shape)
            # forward pass to get outputs
            output_y = net(x,xd,xf)
#            output_y = net(x)

#            print(output_y.shape,py.shape)
            # calculate the loss between predicted and target keypoints
            loss = criterion(output_y, py)
            
            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()
#            if i % 10 == 9:    # print every 10 batches
#                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/10))
#                running_loss = 0.0
        training_loss.append(running_loss/(i+1))

        running_vloss = 0.0        
        with torch.no_grad():
            for i, (x, xd, xf, py) in enumerate(valid_loader):
                py = py.to(device)
                x = x.to(device)
                xf = xf.to(device)
                xd = xd.to(device)
    
                # zero the parameter (weight) gradients
                optimizer.zero_grad()
                
                # print(images.shape)
                # forward pass to get outputs
                output_y = net(x,xd,xf)
    #            output_y = net(x)
    
    #            print(output_y.shape,py.shape)
                # calculate the loss between predicted and target keypoints
                loss = criterion(output_y, py)
    
                # print loss statistics
                running_vloss += loss.item()
    #            if i % 10 == 9:    # print every 10 batches
    #                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/10))
    #                running_loss = 0.0
            valid_loss.append(running_vloss/(i+1))
        
        running_tloss = 0.0        
        with torch.no_grad():
            for i, (x, xd, xf, py) in enumerate(test_loader):
                py = py.to(device)
                x = x.to(device)
                xf = xf.to(device)
                xd = xd.to(device)
    
                # zero the parameter (weight) gradients
                optimizer.zero_grad()
                
                # print(images.shape)
                # forward pass to get outputs
                output_y = net(x,xd,xf)
    #            output_y = net(x)
    
    #            print(output_y.shape,py.shape)
                # calculate the loss between predicted and target keypoints
                loss = criterion(output_y, py)
    
                # print loss statistics
                running_tloss += loss.item()
    #            if i % 10 == 9:    # print every 10 batches
    #                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/10))
    #                running_loss = 0.0
            test_loss.append(running_tloss/(i+1))

    print('Finished Training')
    return training_loss,valid_loss,test_loss
if __name__ == '__main__':
    # train your network
    n_epochs = 100 # start small, and increase when you've decided on your model structure and hyperparams
    path = "./processModelcuda249.pth"
#    net.load_state_dict(torch.load(path))
    print(net.dropout)
    # this is a Workspaces-specific context manager to keep the connection
    # alive while training your model, not part of pytorch
    training_loss,valid_loss,test_loss = train_net(n_epochs)
    
#    font = {'family': 'arial', 'size': 20}
#    fsize = 20
#    plt.figure()
#    plt.plot(np.array(training_loss),label='train',lw=2)
#    plt.plot(np.array(valid_loss),label='valid',lw=2)
#    plt.plot(np.array(test_loss),label='test',lw=2)
#    plt.title('CRNN training history',fontdict=font)
#    plt.legend(loc='upper right',fontsize=fsize)
#    plt.xticks(fontfamily='arial',fontsize=fsize)
#    plt.yticks(fontfamily='arial',fontsize=fsize)
#    plt.xlabel('epoch',fontdict=font)
#    plt.ylabel('MSE',fontdict=font)
#    plt.show()
    
#    start_time = time.time()
#    error = test_net()
#    end_time = time.time()
#    print('time',end_time-start_time)
    
    error,cerror, aerr = test_net_multi(10)
    angerr = np.mean(np.mean(np.mean(aerr,0),0),0)
    stxerr = np.mean(np.mean(np.mean(aerr,0),0),1)
    styerr = np.mean(np.mean(np.mean(aerr,0),1),1)
    
#    print(angerr,stxerr,styerr)
    
#    font = {'family': 'arial','weight':  'bold', 'size': 32}
#    fsize = 32
#    plt.figure(figsize = (12, 9))
#    plt.bar(range(181),angerr,color='#999899')
#    plt.title('MAE in different angles',fontdict=font)
#    plt.xlabel('Angle(°)',fontdict=font)
#    plt.ylabel('MAE(nm)',fontdict=font)
#    plt.xticks(fontfamily='arial',fontsize=fsize,fontweight='bold')
#    plt.yticks(fontfamily='arial',fontsize=fsize,fontweight='bold') 
#    plt.show()
#
#    plt.figure(figsize = (12, 9))
#    plt.bar(range(1,39),stxerr,color='#999899')
#    plt.title('MAE from different baselines',fontdict=font)
#    plt.ylim([0.4,1.05])
#    plt.xlabel('Steps',fontdict=font)
#    plt.ylabel('MAE(nm)',fontdict=font)
#    plt.xticks(fontfamily='arial',fontsize=fsize,fontweight='bold')
#    plt.yticks(fontfamily='arial',fontsize=fsize,fontweight='bold') 
#    plt.show()
#    
#    plt.figure(figsize = (12, 9))
#    plt.bar(range(1,11),styerr,color='#999899')
#    plt.title('MAE for different prediction range',fontdict=font)
#    plt.xlabel('Steps',fontdict=font)
#    plt.ylabel('MAE(nm)',fontdict=font)
#    plt.xticks(fontfamily='arial',fontsize=fsize,fontweight='bold')
#    plt.yticks(fontfamily='arial',fontsize=fsize,fontweight='bold')
#    plt.show()
    
    print(error,cerror)

#    model_train(PeMS, blocks, args)
#    model_test(PeMS, PeMS.get_len('test'), n_his, n_pred, args.inf_mode)
