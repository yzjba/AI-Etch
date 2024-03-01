# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 10:20:20 2023

@author: yzj
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import math
import pandas as pd
import pickle

    
def complot(basedist,pdist,tdist):
    #pim = np.zeros((1000,740,3), dtype="uint8")
    cenpos = [130,99.5]
    
    xb = np.zeros(181)
    yb = np.zeros(181)
    xp = np.zeros(181)
    yp = np.zeros(181)
    xt = np.zeros(181)
    yt = np.zeros(181)
    
    for a in range(-90,91):
        dist = basedist[a+90]
        xb[a+90] = dist*math.cos(a/180*math.pi)+cenpos[0]
        yb[a+90] = dist*math.sin(a/180*math.pi)+cenpos[1]
        #pim[int(np.round(xb[a+90])),int(np.round(yb[a+90])),:] = [0,0,255]
        
        dist = pdist[a+90]
        xp[a+90] = dist*math.cos(a/180*math.pi)+cenpos[0]
        yp[a+90] = dist*math.sin(a/180*math.pi)+cenpos[1]
        #pim[int(np.round(xp[a+90])),int(np.round(yp[a+90]))] = [255,0,0]
        
        dist = tdist[a+90]
        xt[a+90] = dist*math.cos(a/180*math.pi)+cenpos[0]
        yt[a+90] = dist*math.sin(a/180*math.pi)+cenpos[1]
        #pim[int(np.round(xt[a+90])),int(np.round(yt[a+90]))] = [0,255,0]
    
#    xb = 0.27*xb
#    yb = 0.27*yb
#    xp = 0.27*xp
#    yp = 0.27*yp
#    xt = 0.27*xt
#    yt = 0.27*yt
    print(xb.shape,xt.shape)
    ind = np.argmax(xb,axis=0)
    mxb = xb[ind]
    myb = yb[ind]
    ind = np.argmax(xt,axis=0)
    mxt = xt[ind]
    myt = yt[ind]

    font = {'family': 'arial','weight':  'bold', 'size': 24}
    fsize = 24
    plt.figure(figsize = (6, 9))
#    plt.plot(yb,700-xb,'--',label='base',linewidth=3,color='#00BFFF')
    #plt.plot(yt,700-xt,'g',label='truth')
                 
    x = np.linspace(0,200,1000)
    plt.fill_between(x,0,569,color='#999899',alpha=1)
    plt.fill(yb,700-xb,color='#FFFFFF',alpha=1)
             
    plt.plot([50,150],[569,569],'k',linewidth=3,color='#000000')
#    plt.plot([myb,myb],[700-mxb,569],'k',linewidth=3,color='#000000')
    plt.plot([50,150],[700-mxb,700-mxb],'k',linewidth=3,color='#000000')
    s = int(np.round(569+mxb-700))
    plt.text(myb-20,634-mxb/2,str(s)+'nm',fontdict=font)
    plt.annotate(' ', 
             xy=(myb,569),
             xytext=(myb,634-mxb/2),
             arrowprops=dict(facecolor='black', shrink=0.0, width=2.0),
             fontfamily='arial',fontsize=fsize,fontweight='bold')
    plt.annotate(' ', 
             xy=(myb,700-mxb),
             xytext=(myb,634-mxb/2),
             arrowprops=dict(facecolor='black', shrink=0.0, width=2.0),
             fontfamily='arial',fontsize=fsize,fontweight='bold')
    plt.text(40,280,'Si',fontdict=font)
    #plt.title('Etching',fontdict=font)
    plt.xlim([20,180])
    plt.ylim([250,600])
#    plt.xlabel('nm',fontdict=font)
#    plt.ylabel('nm',fontdict=font)
    plt.xticks(range(20,200,40),fontfamily='arial',fontsize=fsize,fontweight='bold')
    plt.yticks(fontfamily='arial',fontsize=fsize,fontweight='bold') 
#    plt.legend(loc='lower right',fontsize=fsize)
    plt.show()

    plt.figure(figsize = (6, 9))
    plt.plot(yb,700-xb,'--',label='base',linewidth=3,color='#00BFFF')
    plt.plot(yp,700-xp,label='predict',linewidth=3,color='#FF6A09',alpha=0.7)
    #plt.plot(yt,700-xt,'g',label='truth')
    x = np.linspace(0,200,1000)
    plt.fill_between(x,0,569,color='#999899',alpha=1)
    plt.fill(yt,700-xt,color='#FFFFFF',alpha=1)
    
    plt.plot([50,150],[569,569],'k',linewidth=3,color='#000000')
    plt.plot([50,150],[700-mxt,700-mxt],'k',linewidth=3,color='#000000')
    s = int(np.round(569+mxt-700))
    plt.text(myt-20,634-mxt/2,str(s)+'nm',fontdict=font)
    plt.annotate(' ', 
             xy=(myt,569),
             xytext=(myt,634-mxt/2),
             arrowprops=dict(facecolor='black', shrink=0.0, width=2.0),
             fontfamily='arial',fontsize=fsize,fontweight='bold')
    plt.annotate(' ', 
             xy=(myt,700-mxt),
             xytext=(myt,634-mxt/2),
             arrowprops=dict(facecolor='black', shrink=0.0, width=2.0),
             fontfamily='arial',fontsize=fsize,fontweight='bold')
    plt.text(40,280,'Si',fontdict=font)
    #plt.title('Etching',fontdict=font)
    plt.xlim([20,180])
    plt.ylim([250,600])
#    plt.xlabel('nm',fontdict=font)
#    plt.ylabel('nm',fontdict=font)
    plt.xticks(range(20,200,40),fontfamily='arial',fontsize=fsize,fontweight='bold')
    plt.yticks(fontfamily='arial',fontsize=fsize,fontweight='bold') 
    plt.legend(loc='lower right',fontsize=fsize)
    plt.show()