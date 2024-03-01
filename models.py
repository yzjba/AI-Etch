# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 11:02:51 2023

@author: yzj
"""

## Define the neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I

class DNN(nn.Module):

    def __init__(self):
        super(DNN, self).__init__()
        
        ## Define all the layers of DNN
        
        # Fully-connected (linear) layers
        self.fc1 = nn.Linear(185, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512,256)
        self.fc4 = nn.Linear(256, 181)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.25)
        
    def forward(self, x,xd,xf):
        ## Define the feedforward behavior of this model
        ## x is the input, xd is the difference, xf is the etching parameters
        
        xff = torch.unsqueeze(xf,1)
#        print(xff.shape)
        xff = xff.repeat((1,x.shape[1],1))
#        print("DNN",x.shape,xd.shape,xff.shape)
        x = torch.cat((x,xd,xff),2)
#        print(x.shape)
        x = x[:,-1,:]
#        print(x.shape)
        x = x[:,181:]
#        x = torch.cat((x,xff),1)
#        print(x.shape)
        
        x = x.view(x.size(0), -1)
        
        # linear layers with dropout in between
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        #x = self.dropout(x)
        x = self.fc4(x)
        
        return x

class RNN(nn.Module):

    def __init__(self):
        super(RNN, self).__init__()
        
        ## Define all the layers of this RNN
                
        hidden_dim = 128
        fdim = 181
        
        # Fully-connected (linear) layers
        self.conv1 = nn.Conv2d(1,fdim,(1,fdim))
        
        self.lstm = nn.LSTM(fdim, hidden_dim, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.fc = nn.Linear(hidden_dim*48, 1024)
        
        self.fc1 = nn.Linear(1024, 512)
        
        self.fc2 = nn.Linear(512,256)
        
        self.fco = nn.Linear(256, fdim)

        # Dropout
        self.dropout = nn.Dropout(p=0.25)
        
        
    def forward(self, x, xd, xf):
        ## Define the feedforward behavior of this model
        ## x is the input, xd is the difference, xf is the etching parameters

        x,(hn,cn) = self.lstm(xd)

#        # Prep for linear layer / Flatten
        #print(x.shape,hn.shape,cn.shape,x.size(0))
        x = x.contiguous().view(x.size(0), -1)
        
        # linear layers with dropout in between
        x = F.sigmoid(self.fc(x))
#        x = torch.cat((x,xf),1)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        #x = self.dropout(x)
        x = self.fco(x)
        
        return x

class CRNN(nn.Module):

    def __init__(self):
        super(CRNN, self).__init__()
        
        ## Define all the layers of this CRNN
        
        hidden_dim = 128
        fdim = 181
        
        # Fully-connected (linear) layers
        self.conv1 = nn.Conv2d(1,32,(1,7))
        self.conv2 = nn.Conv2d(32,64,(1,7))
        self.conv3 = nn.Conv2d(64,128,(1,7))
#        self.conv4 = nn.Conv2d(128,256,(1,7))
        self.conv5 = nn.Conv2d(128,fdim,(1,5))
        
        self.pool = nn.MaxPool2d((1,3), (1,3))
        self.pool2 = nn.MaxPool2d((1,2), (1,2))
        
        self.lstm = nn.LSTM(fdim, hidden_dim, 1,batch_first=True)
        self.lstm1 = nn.LSTM(fdim, hidden_dim, 1,batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.fc = nn.Linear(hidden_dim*48, 1024)
#        self.fc = nn.Linear(181, 1024)
        
        self.fcf = nn.Linear(4,4)
        
        self.fc1 = nn.Linear(1024+4, 512)
        
        self.fc2 = nn.Linear(512+4,256)
        
        self.fc3 = nn.Linear(256+4,256)
        
        self.fc4 = nn.Linear(256+4,256)
        
        self.fco = nn.Linear(256, 181)

        
        # Dropout
        self.dropout = nn.Dropout(p=0.4)
        
        
    def forward(self, x, xd, xf):
        ## Define the feedforward behavior of this model
        ## x is the input, xd is the difference, xf is the etching parameters

        x,(hn,cn) = self.lstm(x)
        xd,(hn,cn) = self.lstm1(xd)

        x = xd.contiguous().view(x.size(0), -1)
        
        # linear layers with dropout in between
        #x = torch.cat((x,xf),1)
#        print(x.shape)
        x = F.sigmoid(self.fc(x))
#        print(x.shape)
        x = torch.cat((x,xf),1)
#        print(x.shape)
        x = F.sigmoid(self.fc1(x))
        x = torch.cat((x,xf),1)
        x = F.sigmoid(self.fc2(x))
        x = torch.cat((x,xf),1)
        x = F.sigmoid(self.fc3(x))
#        x = torch.cat((x,xf),1)
#        x = F.sigmoid(self.fc4(x))
#        x = torch.cat((x,xf),1)
#        x = self.dropout(x)
        x = self.fco(x)
        
        return x   

    
class CCNN(nn.Module):

    def __init__(self):
        super(CCNN, self).__init__()
        
        ## Define all the layers of this CCNN.
        ## x is the input, xd is the difference, xf is the etching parameters
        
        hidden_dim = 128
        fdim = 181

        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1,32,(1,7))
        self.conv2 = nn.Conv2d(32,64,(1,7))
        self.conv3 = nn.Conv2d(64,128,(1,7))
        self.conv4 = nn.Conv2d(128,128,(1,7))
        self.conv5 = nn.Conv2d(128,fdim,(1,5))
        
        self.pool = nn.MaxPool2d((1,3), (1,3))
        self.pool2 = nn.MaxPool2d((1,2), (1,2))
        
        self.lstm = nn.LSTM(fdim, hidden_dim, 1,batch_first=True)
        self.lstm1 = nn.LSTM(fdim, hidden_dim, 1,batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.fc = nn.Linear(48*fdim, 1024)
#        self.fc = nn.Linear(181, 1024)
        
        self.fcf = nn.Linear(4,4)
        
        self.fc1 = nn.Linear(1024+4, 512)
        
        self.fc2 = nn.Linear(512+4,256)
        
        self.fc3 = nn.Linear(256+4,256)
        
        self.fc4 = nn.Linear(256+4,256)
        
        self.fco = nn.Linear(256, 181)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.4)
        
        
    def forward(self, x, xd, xf):
        ## Define the feedforward behavior of this model
        ## x is the input, xd is the difference, xf is the etching parameters

        x = torch.unsqueeze(xd,1)
#        print(x.shape)
        x = F.relu(self.pool(self.conv1(x)))
#        print(x.shape)
        x = F.relu(self.pool(self.conv2(x)))
#        print(x.shape)
        x = F.relu(self.pool2(self.conv3(x)))
#        print(x.shape)
#        x = F.relu(self.pool(self.conv4(x)))
        x = F.relu(self.conv5(x))
#        print('conv',x.shape)
        x = torch.squeeze(x,3)

        x = x.contiguous().view(x.size(0), -1)
        
        # linear layers with dropout in between
        #x = torch.cat((x,xf),1)
#        print(x.shape)
        x = F.sigmoid(self.fc(x))
#        print(x.shape)
        x = torch.cat((x,xf),1)
#        print(x.shape)
        x = F.sigmoid(self.fc1(x))
        x = torch.cat((x,xf),1)
        x = F.sigmoid(self.fc2(x))
        x = torch.cat((x,xf),1)
        x = F.sigmoid(self.fc3(x))
#        x = torch.cat((x,xf),1)
#        x = F.sigmoid(self.fc4(x))
#        x = torch.cat((x,xf),1)
#        x = self.dropout(x)
        x = self.fco(x)
        
        return x
    
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        
        ## Define all the layers of this CNN        
        
        hidden_dim = 128
        fdim = 181

        # Convolutional layers
        self.conv1 = nn.Conv2d(1,32,(1,7))
        self.conv2 = nn.Conv2d(32,64,(1,7))
        self.conv3 = nn.Conv2d(64,128,(1,7))
        self.conv4 = nn.Conv2d(128,128,(1,7))
        self.conv5 = nn.Conv2d(128,fdim,(1,5))
        
        self.pool = nn.MaxPool2d((1,3), (1,3))
        self.pool2 = nn.MaxPool2d((1,2), (1,2))
        
        self.lstm = nn.LSTM(fdim, hidden_dim, 1,batch_first=True)
        self.lstm1 = nn.LSTM(fdim, hidden_dim, 1,batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.fc = nn.Linear(48*fdim, 1024)
#        self.fc = nn.Linear(181, 1024)
        
        self.fcf = nn.Linear(4,4)
        
        self.fc1 = nn.Linear(1024+4, 512)
        
        self.fc2 = nn.Linear(512+4,256)
        
        self.fc3 = nn.Linear(256+4,256)
        
        self.fc4 = nn.Linear(256,256)
        
        self.fco = nn.Linear(256, 181)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.4)
        
        
    def forward(self, x, xd, xf):
        ## Define the feedforward behavior of this model
        ## x is the input, xd is the difference, xf is the etching parameters

#        print(x.shape)
        x = torch.unsqueeze(x,1)
#        print(x.shape)
        x = F.relu(self.pool(self.conv1(x)))
#        print(x.shape)
        x = F.relu(self.pool(self.conv2(x)))
#        print(x.shape)
        x = F.relu(self.pool2(self.conv3(x)))
#        print(x.shape)
#        x = F.relu(self.pool(self.conv4(x)))
        x = F.relu(self.conv5(x))
#        print('conv',x.shape)
        x = torch.squeeze(x,3)

        x = x.contiguous().view(x.size(0), -1)
        
        # linear layers with dropout in between
        #x = torch.cat((x,xf),1)
#        print(x.shape)
        x = F.sigmoid(self.fc(x))
#        print(x.shape)
        x = torch.cat((x,xf),1)
#        print(x.shape)
        x = F.sigmoid(self.fc1(x))
        x = torch.cat((x,xf),1)
        x = F.sigmoid(self.fc2(x))
        x = torch.cat((x,xf),1)
        x = F.sigmoid(self.fc3(x))
#        x = torch.cat((x,xf),1)
#        x = F.sigmoid(self.fc4(x))
#        x = torch.cat((x,xf),1)
#        x = self.dropout(x)
        x = self.fco(x)
        
        return x 