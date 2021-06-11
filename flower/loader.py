#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:32:22 2020

@author: aaronli
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch import nn
from torch.utils.data import DataLoader
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
import pickle

# Detection
class waveformDetectionDL(Dataset):
    
    def __init__(self, path):
        
        # Directory of normal and abnormal data
        normal_path = path + 'Normal/'
        abnormal_path = path + 'Abnormal/'
        
        normal_files = [f for f in os.listdir(normal_path) if f.endswith('.csv')]
        abnormal_files = [f for f in os.listdir(abnormal_path) if f.endswith('.csv')]
        
        np_normal = []
        np_abnormal = []
        
        for f in normal_files:
            temp_file = np.loadtxt(normal_path + f, delimiter=',', dtype=np.float32)
            np_normal.append(temp_file)
        
        for f in abnormal_files:
            temp_file = np.loadtxt(abnormal_path + f, delimiter=',', dtype=np.float32)
            np_abnormal.append(temp_file)
            
        # x: input data, y: target
        self.y_data = np.concatenate([np.zeros(len(np_normal),dtype=np.long), \
                                      np.zeros(len(np_abnormal),dtype=np.long)+1])
        self.x_data = np.concatenate([np_normal, np_abnormal])
        self.len = self.x_data.shape[0]
        self.x_data = torch.from_numpy(self.x_data)
        self.y_data = torch.from_numpy(self.y_data)
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len
    
class WaveformDetectionDLPickle(Dataset):
    
    def __init__(self, path, client_num):
        
        # Directory of normal and abnormal data
        if client_num == -1:
            with open(os.path.join(path, 'test_data.pkl'), 'rb') as f:
                data = pickle.load(f)
        else:
            with open(os.path.join(path, f'data_{client_num}.pkl'), 'rb') as f:
                data = pickle.load(f)
        self.x_data = data['x_data']
        self.y_data = data['y_data']
        self.len = self.x_data.shape[0]
        self.x_data = torch.from_numpy(self.x_data)
        self.y_data = torch.from_numpy(self.y_data)
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len
    
class waveformDetectionFeature(Dataset):
    def __init__(self, path):
        
        # Directory of normal and abnormal data
        normal_path = path + 'Normal/'
        abnormal_path = path + 'Abnormal/'
        
        normal_files = [f for f in os.listdir(normal_path) if f.endswith('.csv')]
        abnormal_files = [f for f in os.listdir(abnormal_path) if f.endswith('.csv')]
        
        np_normal = np.empty([0,18])
        np_abnormal = np.empty([0,18])
        
        for f in normal_files:
            temp_file = np.loadtxt(normal_path + f, delimiter=',', dtype=np.float32)
            np_normal = np.append(np_normal, temp_file, axis=0)
        
        for f in abnormal_files:
            temp_file = np.loadtxt(abnormal_path + f, delimiter=',', dtype=np.float32)
            np_abnormal = np.append(np_abnormal, temp_file, axis=0)
            
        # x: input data, y: target
        self.y_data = np.concatenate([np.zeros(np_normal.shape[0],dtype=np.long), \
                                      np.zeros(np_abnormal.shape[0],dtype=np.long)+1])
        self.x_data = np.concatenate([np_normal, np_abnormal])
        self.len = self.x_data.shape[0]
        self.x_data = torch.from_numpy(self.x_data)
        self.y_data = torch.from_numpy(self.y_data)
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len
    
    
# Diagnosis 
class waveformDiagnosisDL(Dataset):
    
    def __init__(self, path):
        
        # Directory of normal and abnormal data
        normal_path = path + 'Normal/'
        fault_1_path = path + 'Fault_1/'
        fault_2_path = path + 'Fault_2/'
        fault_3_path = path + 'Fault_3/'
        fault_4_path = path + 'Fault_4/'
        fault_5_path = path + 'Fault_5/'
        fault_6_path = path + 'Fault_6/'
        fault_7_path = path + 'Fault_7/'
        fault_8_path = path + 'Fault_8/'
        
        
        normal_files = [f for f in os.listdir(normal_path) if f.endswith('.csv')]
        fault_1_files = [f for f in os.listdir(fault_1_path) if f.endswith('.csv')]
        fault_2_files = [f for f in os.listdir(fault_2_path) if f.endswith('.csv')]
        fault_3_files = [f for f in os.listdir(fault_3_path) if f.endswith('.csv')]
        fault_4_files = [f for f in os.listdir(fault_4_path) if f.endswith('.csv')]
        fault_5_files = [f for f in os.listdir(fault_5_path) if f.endswith('.csv')]
        fault_6_files = [f for f in os.listdir(fault_6_path) if f.endswith('.csv')]
        fault_7_files = [f for f in os.listdir(fault_7_path) if f.endswith('.csv')]
        fault_8_files = [f for f in os.listdir(fault_8_path) if f.endswith('.csv')]
        
        np_normal = []
        np_fault_1 = []
        np_fault_2 = []
        np_fault_3 = []
        np_fault_4 = []
        np_fault_5 = []
        np_fault_6 = []
        np_fault_7 = []
        np_fault_8 = []
        
        
        for f in normal_files:
            temp_file = np.loadtxt(normal_path + f, delimiter=',', dtype=np.float32)
            np_normal.append(temp_file)
        
        for f in fault_1_files:
            temp_file = np.loadtxt(fault_1_path + f, delimiter=',', dtype=np.float32)
            np_fault_1.append(temp_file)
            
        for f in fault_2_files:
            temp_file = np.loadtxt(fault_2_path + f, delimiter=',', dtype=np.float32)
            np_fault_2.append(temp_file)
            
        for f in fault_3_files:
            temp_file = np.loadtxt(fault_3_path + f, delimiter=',', dtype=np.float32)
            np_fault_3.append(temp_file)
            
        for f in fault_4_files:
            temp_file = np.loadtxt(fault_4_path + f, delimiter=',', dtype=np.float32)
            np_fault_4.append(temp_file)
            
        for f in fault_5_files:
            temp_file = np.loadtxt(fault_5_path + f, delimiter=',', dtype=np.float32)
            np_fault_5.append(temp_file)
            
        for f in fault_6_files:
            temp_file = np.loadtxt(fault_6_path + f, delimiter=',', dtype=np.float32)
            np_fault_6.append(temp_file)
            
        for f in fault_7_files:
            temp_file = np.loadtxt(fault_7_path + f, delimiter=',', dtype=np.float32)
            np_fault_7.append(temp_file)
            
        for f in fault_8_files:
            temp_file = np.loadtxt(fault_8_path + f, delimiter=',', dtype=np.float32)
            np_fault_8.append(temp_file)
            
        # x: input data, y: target
        self.y_data = np.concatenate([np.zeros(len(np_normal),dtype=np.long), \
                                      np.zeros(len(np_fault_1),dtype=np.long)+1, \
                                      np.zeros(len(np_fault_2),dtype=np.long)+2, \
                                      np.zeros(len(np_fault_3),dtype=np.long)+3, \
                                      np.zeros(len(np_fault_4),dtype=np.long)+4, \
                                      np.zeros(len(np_fault_5),dtype=np.long)+5, \
                                      np.zeros(len(np_fault_6),dtype=np.long)+6, \
                                      np.zeros(len(np_fault_7),dtype=np.long)+7, \
                                      np.zeros(len(np_fault_8),dtype=np.long)+8, ])
        self.x_data = np.concatenate([np_normal, np_fault_1, np_fault_2, np_fault_3, np_fault_4, 
                                      np_fault_5, np_fault_6, np_fault_7, np_fault_8])
        self.len = self.x_data.shape[0]
        self.x_data = torch.from_numpy(self.x_data)
        self.y_data = torch.from_numpy(self.y_data)
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

class waveformDiagnosisFeature(Dataset):
    
    def __init__(self, path):
        
        # Directory of normal and abnormal data
        normal_path = path + 'Normal/'
        fault_1_path = path + 'Fault_1/'
        fault_2_path = path + 'Fault_2/'
        fault_3_path = path + 'Fault_3/'
        fault_4_path = path + 'Fault_4/'
        fault_5_path = path + 'Fault_5/'
        fault_6_path = path + 'Fault_6/'
        fault_7_path = path + 'Fault_7/'
        fault_8_path = path + 'Fault_8/'
        
        
        normal_files = [f for f in os.listdir(normal_path) if f.endswith('.csv')]
        fault_1_files = [f for f in os.listdir(fault_1_path) if f.endswith('.csv')]
        fault_2_files = [f for f in os.listdir(fault_2_path) if f.endswith('.csv')]
        fault_3_files = [f for f in os.listdir(fault_3_path) if f.endswith('.csv')]
        fault_4_files = [f for f in os.listdir(fault_4_path) if f.endswith('.csv')]
        fault_5_files = [f for f in os.listdir(fault_5_path) if f.endswith('.csv')]
        fault_6_files = [f for f in os.listdir(fault_6_path) if f.endswith('.csv')]
        fault_7_files = [f for f in os.listdir(fault_7_path) if f.endswith('.csv')]
        fault_8_files = [f for f in os.listdir(fault_8_path) if f.endswith('.csv')]
        
        np_normal = np.empty([0,18])
        np_fault_1 = np.empty([0,18])
        np_fault_2 = np.empty([0,18])
        np_fault_3 = np.empty([0,18])
        np_fault_4 = np.empty([0,18])
        np_fault_5 = np.empty([0,18])
        np_fault_6 = np.empty([0,18])
        np_fault_7 = np.empty([0,18])
        np_fault_8 = np.empty([0,18])
        
        
        for f in normal_files:
            temp_file = np.loadtxt(normal_path + f, delimiter=',', dtype=np.float32)
            np_normal = np.append(np_normal, temp_file, axis=0)
        
        for f in fault_1_files:
            temp_file = np.loadtxt(fault_1_path + f, delimiter=',', dtype=np.float32)
            np_fault_1 = np.append(np_fault_1, temp_file, axis=0)
            
        for f in fault_2_files:
            temp_file = np.loadtxt(fault_2_path + f, delimiter=',', dtype=np.float32)
            np_fault_2 = np.append(np_fault_1, temp_file, axis=0)
            
        for f in fault_3_files:
            temp_file = np.loadtxt(fault_3_path + f, delimiter=',', dtype=np.float32)
            np_fault_3 = np.append(np_fault_1, temp_file, axis=0)
            
        for f in fault_4_files:
            temp_file = np.loadtxt(fault_4_path + f, delimiter=',', dtype=np.float32)
            np_fault_4 = np.append(np_fault_1, temp_file, axis=0)
            
        for f in fault_5_files:
            temp_file = np.loadtxt(fault_5_path + f, delimiter=',', dtype=np.float32)
            np_fault_5 = np.append(np_fault_1, temp_file, axis=0)
            
        for f in fault_6_files:
            temp_file = np.loadtxt(fault_6_path + f, delimiter=',', dtype=np.float32)
            np_fault_6 = np.append(np_fault_1, temp_file, axis=0)
            
        for f in fault_7_files:
            temp_file = np.loadtxt(fault_7_path + f, delimiter=',', dtype=np.float32)
            np_fault_7 = np.append(np_fault_1, temp_file, axis=0)
            
        for f in fault_8_files:
            temp_file = np.loadtxt(fault_8_path + f, delimiter=',', dtype=np.float32)
            np_fault_8 = np.append(np_fault_1, temp_file, axis=0)
            
        # x: input data, y: target
        self.y_data = np.concatenate([np.zeros(np_normal.shape[0] ,dtype=np.long), \
                                      np.zeros(np_fault_1.shape[0] ,dtype=np.long)+1, \
                                      np.zeros(np_fault_2.shape[0] ,dtype=np.long)+2, \
                                      np.zeros(np_fault_3.shape[0] ,dtype=np.long)+3, \
                                      np.zeros(np_fault_4.shape[0] ,dtype=np.long)+4, \
                                      np.zeros(np_fault_5.shape[0] ,dtype=np.long)+5, \
                                      np.zeros(np_fault_6.shape[0] ,dtype=np.long)+6, \
                                      np.zeros(np_fault_7.shape[0] ,dtype=np.long)+7, \
                                      np.zeros(np_fault_8.shape[0] ,dtype=np.long)+8, ])
        self.x_data = np.concatenate([np_normal, np_fault_1, np_fault_2, np_fault_3, np_fault_4, 
                                      np_fault_5, np_fault_6, np_fault_7, np_fault_8])
        self.len = self.x_data.shape[0]
        self.x_data = torch.from_numpy(self.x_data)
        self.y_data = torch.from_numpy(self.y_data)
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len
