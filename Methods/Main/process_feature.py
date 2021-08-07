# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:17:32 2020
modified on Mon Jan 4 20点05分 2021
modified on Mon Feb 03 17:16 2021
@author: xinxi
"""

import numpy as np
import os
import sys
import torch



seed = 1
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def getFeature(opt, traiY, row_sim_matrix, col_sim_matrix):
    seed = 1
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    feature_type = opt.feature_type
    if feature_type == 'one_hot':
        opt.in_features = traiY.shape[0] + traiY.shape[1]
        feat_array = np.eye(opt.in_features)
    elif feature_type == 'random_uniform':
        np.random.seed(seed)
        opt.in_features = traiY.shape[0] + traiY.shape[1]
        feat_array = np.random.rand(traiY.shape[0]+traiY.shape[1], opt.in_features)
    elif feature_type == 'random_normal':
        np.random.seed(seed)
        opt.in_features = traiY.shape[0] + traiY.shape[1]
        feat_array = np.random.randn(traiY.shape[0]+traiY.shape[1], opt.in_features)
    elif feature_type == 'dgi':
        from process_dgi import dgi_embed
        opt.in_features = traiY.shape[0] + traiY.shape[1]
        # opt.in_features = opt.k_dim
        embedding_file = opt.dataPrefix+'DGI_embedding/'
        if not os.path.exists(embedding_file):
            os.makedirs(embedding_file)
        output_file = embedding_file+'dgi_dim'+str(opt.in_features)+'_'+str(opt.nfold)+'nfold_'+'kfold'+str(opt.kfold)+'.txt'
        if not os.path.exists(output_file):            
            association = np.vstack((np.hstack((np.zeros((traiY.shape[0],traiY.shape[0]),dtype=np.float32),
                                                traiY)),
                                     np.hstack((traiY.T,
                                                np.zeros((traiY.shape[1],traiY.shape[1]),dtype=np.float32)
                                                ))
                                     ))
            if not os.path.exists(opt.dataPrefix+'dgi_model/'):
                os.mkdir(opt.dataPrefix+'dgi_model/')
            feat_array = dgi_embed(association, np.eye(opt.in_features), opt.in_features,
                                   opt.dataPrefix + '/dgi_model/best_dgi_in_features'+str(opt.in_features)+'_'+str(opt.nfold)+'nfold_'+'kfold'+str(opt.kfold)+'.pkl')
            np.savetxt(output_file, feat_array)
        else:
            feat_array = np.loadtxt(output_file)
    elif feature_type.startswith('bionev'):
        opt.in_features = opt.k_dim
        method = feature_type.split('_')[-1]
        embedding_file = opt.dataPrefix+'bio_embedding/'
        if not os.path.exists(embedding_file):
            os.makedirs(embedding_file)
        output_file = embedding_file+method+'_dim'+str(opt.in_features)+'_'+str(opt.nfold)+'nfold'+'_kfold'+str(opt.kfold)+'.txt'
        
        class Args(object):
            def __init__(self):
                return
        args = Args()
        args.input = opt.dataPrefix + 'tra_kfold' + str(opt.kfold) + '_seed' + str(opt.seed) + '_total.txt'
        args.task = 'none'
        args.output = output_file
        args.testingratio = 0.2
        args.number_walks = 32
        args.walk_length = 64
        args.workers = 1
        args.dimensions = opt.k_dim
        args.window_size = 10
        args.epochs = 5
        args.p = 1.
        args.q = 1.
        args.method = method
        args.lable_file = ''
        args.negative_ratio = 5
        args.weighted = False
        args.directed = False
        args.order = 2
        args.weight_decay = 5e-4
        args.kstep = 4
        args.lr = 0.01
        args.alpha = 0.3
        args.beta = 0
        args.nu1 = 1e-5
        args.nu2 = 1e-4
        args.bs = 200
        args.encoder_list = '[1000, '+str(args.dimensions)+']'
        args.OPT1 = True
        args.OPT2 = True
        args.OPT3 = True
        args.until_layer = 6
        args.dropout = 0
        args.hidden = 32
        args.gae_model_selection = 'gcn_ae'
        args.eval_result_file = None
        args.seed = opt.seed
        if not os.path.exists(output_file):
            from bionev.main import main as BioNEV_main
            BioNEV_main(args)
        feature = np.loadtxt(output_file, skiprows=1)
        feat_dict = dict()
        for item in feature:
            feat_dict[item[0]]=item[1:]
        feat_array = np.zeros((opt.row_num + opt.col_num, len(item[1:])))
        for i in range(opt.row_num + opt.col_num):
            if i not in feat_dict.keys():
                feat_array[i] = np.random.randn(len(item[1:]))
            else:
                feat_array[i] = feat_dict[i]
        
    ### normalize ###
    if opt.normalize == 'col_mean_zero':
        from sklearn import preprocessing
        feat_array = preprocessing.scale(feat_array)
    elif opt.normalize == 'minmax':
        from sklearn.preprocessing import minmax_scale
        feat_array = minmax_scale(feat_array)
    elif opt.normalize == 'softmax':
        from utils.normalization import normalizeSoft
        feat_array = normalizeSoft(feat_array)
    elif opt.normalize == 'row_sum_one':
        from utils.normalization import normalizeRow
        feat_array = normalizeRow(feat_array)
    elif opt.normalize == 'none':
        pass
    else:
        print('parameter opt.normalize is wrong')
        print(opt.normalize)
        sys.exit()
    F_u_temp = feat_array[: traiY.shape[0]]
    F_i_temp = feat_array[traiY.shape[0]: ]
    return np.array(F_u_temp), np.array(F_i_temp)