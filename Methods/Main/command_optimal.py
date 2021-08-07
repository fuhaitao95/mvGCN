#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 19:29:23 2021

@author: fht
"""



import os
import sys

import argparse
from main_func import call_main

parser=argparse.ArgumentParser(description='Bipartite graph prediction by multi-similarity fusion')

parser.add_argument('--dataName', default='ZhangMDA', 
                        choices=['ZhangMDA', 'LiMDA', 
                                 'LiangDDA', 'ZhangDDA', 
                                 'LuoDTI', 'LiuDTI'],
                        help='the data name')
parser.add_argument('--exp_name', default='k_dim',
                    choices=['k_dim','num_layer', 'alp_beta',
                             'similarity', 'layer_no', 'ind_sim', 'feature_type',
                             'caseStudy', 'tsne_flag', 'optimal'])
parser.add_argument('--cross', type=int, default=1, help='cross validate or not')
parser.add_argument('--single_fold', type=int, default=1, help='whether one fold')
parser.add_argument('--kfold',type=int,default=0,help='index of train or validation k number')
parser.add_argument('--test', type=int, default=1, help='test or not')
parser.add_argument('--patience', type=int, default=10, help='patience')
parser.add_argument('--epochs', type=int, default=100, help='epochs')
parser.add_argument('--batch_size', type=int, default=128, help='epochs')


opt = parser.parse_args()
opt.tra_fold = opt.kfold

para_dict = dict()

opt.nfold = 5

opt.prin = 0
opt.result_key = opt.exp_name

opt.hh_fusion = 'add'
opt.layer_att = 0
opt.train_type = 'pretrain'
opt.dgi_weight = 0.
opt.normalize = 'row_sum_one'
opt.init = 'he_normal'
opt.func = 'raw'
opt.score_type = 'cat'
opt.sigmoid_flag = 1
opt.lossType = 'MF_all'
opt.lr = 0.0005
opt.weight_decay = 0.
opt.dropout = 0.1
opt.device = 'cpu'
# opt.epochs = 100
# opt.patience = 10
opt.seed = 1
# opt.batch_size = 128
opt.fastmode = False
# opt.k_dim = 8
# opt.num_layer = 2
# opt.alp = 0.5
# opt.beta = 0.5


# opt.similarity_att = 9
# opt.layer_no = -1
# opt.ind_sim = -1

# opt.feature_type = 'dgi'
# opt.dataName = 'ZhangMDA'



similarity_att_ls = [9]
layer_no_ls = [-1]
ind_sim_ls = [-1]
feature_type_ls = ['dgi']
opt.caseStudy=0
case_x_ls=[-1]
case_y_ls=[-1]
opt.tsne_flag=0



if opt.dataName=='ZhangMDA':
    k_dim_ls = [16]
    num_layer_ls = [4]
    alp_ls = [0.8]
    beta_ls = [0.7]
elif opt.dataName=='LiMDA':
    k_dim_ls = [256]
    num_layer_ls = [2]
    alp_ls = [0.8]
    beta_ls = [0.6]
elif opt.dataName=='LiangDDA':
    k_dim_ls = [256]
    num_layer_ls = [5]
    alp_ls = [0.9]
    beta_ls = [0.9]
elif opt.dataName=='ZhangDDA':
    k_dim_ls = [256]
    num_layer_ls = [1]
    alp_ls = [0.1]
    beta_ls = [0.4]
elif opt.dataName=='LuoDTI':
    k_dim_ls = [256]
    num_layer_ls = [5]
    alp_ls = [0.8]
    beta_ls = [0.7]
elif opt.dataName=='LiuDTI':
    k_dim_ls = [128]
    num_layer_ls = [4]
    alp_ls = [0.3]
    beta_ls = [0.8]
else:
    print('data name is wrong')
    sys.exit(1)



if opt.exp_name=='k_dim':
    k_dim_ls = [8,16,32,64,128,256]
    # k_dim_ls = [8]
elif opt.exp_name=='num_layer':
    num_layer_ls = [1,2,3,4,5]
elif opt.exp_name=='alp_beta':
    alp_ls = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    beta_ls = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
elif opt.exp_name=='similarity':
    similarity_att_ls = [-1, 9]
elif opt.exp_name=='layer_no':
    num_layer_ls = [5]
    layer_no_ls = [0,1,2,3,4,5]
elif opt.exp_name=='ind_sim':
    if opt.dataName in ['ZhangMDA', 'LiMDA', 'LiuDTI']:
        ind_sim_ls = list(range(-1,4))
    elif opt.dataName in ['LiangDDA']:
        ind_sim_ls = list(range(-1,8))
    elif opt.dataName in ['LuoDTI']:
        ind_sim_ls = list(range(-1,20))
    elif opt.dataName in ['ZhangDDA']:
        ind_sim_ls = list(range(-1,12))
    else:
        print('the data doesn"t need ind_sim experiment')
        sys.exit(2)
elif opt.exp_name=='feature_type':
    feature_type_ls = ['one_hot', 'random_uniform', 'random_normal',
                    'dgi',                        
                    'bionev_LINE', 'bionev_SDNE', 'bionev_GAE',
                    'bionev_DeepWalk', 'bionev_node2vec', 'bionev_struc2vec', 
                    'bionev_Laplacian', 'bionev_GF', 'bionev_SVD', 'bionev_HOPE', 'bionev_GraRep']
    # feature_type_ls = ['random_normal']
elif opt.exp_name=='caseStudy':
    opt.caseStudy=1
    if opt.dataName == 'ZhangMDA':
        case_x_ls=[-1,-1,-1]
        case_y_ls=[2, 34, 21]
    elif opt.dataName=='ZhangDDA':
        case_x_ls=[81,-1]
        case_y_ls=[-1,12]
elif opt.exp_name=='tsne_flag':
    opt.tsne_flag=1
elif opt.exp_name=='optimal':
    pass
else:
    print('experiment is wrong!')
    sys.exit(1)



i=1
for k_dim in k_dim_ls:
    for num_layer in num_layer_ls:
        for alp in alp_ls:
            for beta in beta_ls:
                
                for similarity_att in similarity_att_ls:
                    for layer_no in layer_no_ls:
                        for ind_sim in ind_sim_ls:
                            for feature_type in feature_type_ls:
                                for case_ind in range(len(case_x_ls)):
                                    opt.k_dim = k_dim
                                    opt.num_layer = num_layer
                                    opt.alp = alp
                                    opt.beta = beta
                                    
                                    
                                    opt.similarity_att = similarity_att
                                    opt.layer_no = layer_no
                                    opt.ind_sim = ind_sim
                                    
                                    opt.feature_type = feature_type
                                    opt.case_x = case_x_ls[case_ind]
                                    opt.case_y = case_y_ls[case_ind]

                                    call_main(opt)
                                    print(i)
                                    i=i+1
