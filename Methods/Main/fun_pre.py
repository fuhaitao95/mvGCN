#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 03:12:49 2021

@author: fht
"""



import numpy as np
import os


from process_data import read_data


def prepare(opt):
    seed = opt.seed
    kfold = opt.kfold
    nfold = opt.nfold
    dataName = opt.dataName
    
    dataPrefix = '../../Datasets/'+dataName+'/split_data_tra_val_'+str(nfold)+'nfold/'
    opt.dataPrefix = dataPrefix
    
    if not os.path.exists(dataPrefix):
        from process_data import splitDataMain
        splitDataMain(nfold, dataName)
        print('Split data')
    
    dataY, sim_A, sim_b, ANet, bNet, names = read_data(dataName, '../../Datasets/'+dataName+'/'+'used_data/')
    opt.row_num, opt.col_num = sim_A[0].shape[0], sim_b[0].shape[0]
    
    tra_list_cross, val_list_cross = [], []
    for kfold in range(opt.nfold):
        tra_name = dataPrefix+'tra_kfold'+str(kfold)+'_seed' + str(seed) + '.txt'
        val_name = dataPrefix+'val_kfold'+str(kfold)+'_seed' + str(seed) + '.txt'
        
        tra_list_k = np.loadtxt(tra_name, delimiter=',')
        val_list_k = np.loadtxt(val_name, delimiter=',')
        
        tra_list_cross.append(tra_list_k)
        val_list_cross.append(val_list_k)
    
    tra_list_test = np.loadtxt(dataPrefix+'tra_val_seed'+str(seed)+'.txt', delimiter=',')
    tes_list_test = np.loadtxt(dataPrefix+'tes_seed'+str(seed)+'.txt', delimiter=',')
    return sim_A, sim_b, tra_list_cross, val_list_cross, tra_list_test, tes_list_test

        
        