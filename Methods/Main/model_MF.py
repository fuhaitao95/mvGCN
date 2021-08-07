# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 08:30:33 2020
Modified On Tue Jan 5 09点38分 2021
Modified On Tue Jan 13 15点48分 2021
@author: xinxi
"""












import numpy as np

import os
import sys
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_, kaiming_normal_, kaiming_uniform_
import torch as t

from process_dgi import dgi_init, dgi_embed
from utils.weight_inits import *
from NIPLayer import NIP



seed = 1
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def lossF(opt, predictions, targets):
    lossType=opt.lossType
    
    if lossType == 'cross_entropy':
        pos_weight = 1.
        neg_weight = 1.
        weightTensor = torch.zeros(len(targets))
        weightTensor[targets == 1] = pos_weight
        weightTensor[targets == 0] = neg_weight
        if (predictions.min() < 0) | (predictions.max() > 1):
            losses = F.binary_cross_entropy_with_logits(predictions.double(), targets.double(), weight=weightTensor)
        else:
            losses = F.binary_cross_entropy(predictions.double(), targets.double(), weight=weightTensor)
    elif lossType == 'MF_all':
        losses = torch.pow((predictions - targets), 2).mean()
    elif lossType == 'MSE':
        losses = F.mse_loss(predictions, targets)    
    return losses
class MF(torch.nn.Module):
    def __init__(self, opt, traiY):
        super(MF, self).__init__()
        self.row_num, self.col_num = opt.row_num, opt.col_num
        self.k_dim = k_dim = opt.k_dim
        self.dropout = dropout = opt.dropout
        self.opt = opt
        
        if opt.cuda:
            torch.cuda.manual_seed(opt.seed)
            torch.cuda.manual_seed_all(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        
        # dropout for E_0
        self.hidden_dropout0 = torch.nn.Dropout(dropout)
        # NIPlayer message passing
        self.NIPLayer_ls = []
        for i_layer in range(opt.num_layer):
            self.NIPLayer_ls.append([])
            for j_num_sim in range(opt.num_sim):
                temp = NIP(k_dim, k_dim, opt.init, opt.func, dropout)
                self.NIPLayer_ls[i_layer].append(temp)                
        # decoder MLP
        self.decoder0 = nn.Linear(k_dim * 2, k_dim)
        self.decoder1 = nn.Linear(k_dim, int(k_dim/2))
        self.decoder2 = nn.Linear(int(k_dim/2), 1)
        
        self.set_para(opt, k_dim)
        
        self.traiY = traiY
        self.dgi = dgi_init(np.eye(opt.in_features), opt.in_features)
        
        association = np.vstack((np.hstack((np.zeros((traiY.shape[0],traiY.shape[0]),dtype=np.float32), traiY)),
                                 np.hstack((traiY.T, np.zeros((traiY.shape[1],traiY.shape[1]),dtype=np.float32) ))
                                ))
        self.association = association
        if opt.feature_type == 'dgi':
            if opt.train_type == 'finetuning':                
                path = opt.dataPrefix + '/dgi_model/best_dgi_in_features'+str(opt.in_features)+'_'+str(opt.nfold)+'nfold_'+'kfold'+str(opt.kfold)+'.pkl'
                self.dgi.load_state_dict(torch.load(path))
        
    def set_para(self, opt, k_dim):
        if opt.cuda:
            torch.cuda.manual_seed(opt.seed)
            torch.cuda.manual_seed_all(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        # the 0th layer feature transformation parameters
        self.U = Parameter(torch.FloatTensor(opt.num_sim, opt.in_features, k_dim))
        self.V = Parameter(torch.FloatTensor(opt.num_sim, opt.in_features, k_dim))           
        # Huang att parameters
        self.att_embed = Parameter(t.FloatTensor(np.random.rand(opt.num_layer)))        
        self.att_weightX1 = Parameter(t.FloatTensor(k_dim, opt.att_dim))
        self.att_weightY1 = Parameter(t.FloatTensor(k_dim, opt.att_dim))        
        self.att_weightX2 = Parameter(t.FloatTensor(opt.att_dim, 1))        
        self.att_weightY2 = Parameter(t.FloatTensor(opt.att_dim, 1))
        # 2 for cross end att Zeng attention parameters
        self.b_w = Parameter(torch.zeros(k_dim,))
        self.w_w = Parameter(torch.FloatTensor(2*k_dim, k_dim))
        self.linear_att = nn.Linear(k_dim*2, k_dim)
        head = 3
        self.h_w = Parameter(torch.FloatTensor(k_dim, head))
        self.head_linear = torch.nn.Linear(k_dim*head*2, k_dim*2)   
        self.h_t = Parameter(torch.FloatTensor(2*k_dim, 2 * k_dim))      
        # 4
        self.att_vec_X = Parameter(torch.FloatTensor(2*k_dim,3))
        self.att_vec_Y = Parameter(torch.FloatTensor(2*k_dim,3))
        # A3NCF attention
        self.att_cat_X1 = nn.Linear(k_dim * opt.num_sim, 1)
        self.att_cat_X2 = nn.Linear(1, k_dim)
        self.att_cat_Y1 = nn.Linear(k_dim * opt.num_sim, 1)
        self.att_cat_Y2 = nn.Linear(1, k_dim)
        # local globle
        self.att_aff_X1 = nn.Linear(k_dim, int(k_dim/2))
        self.att_aff_X2 = nn.Linear(int(k_dim/2), k_dim)
        self.att_aff_Y1 = nn.Linear(k_dim, int(k_dim/2))
        self.att_aff_Y2 = nn.Linear(int(k_dim/2), k_dim)
        # 直接权重softmax对应求和
        self.x_att = Parameter(torch.FloatTensor(np.zeros(opt.num_sim,)))
        self.y_att = Parameter(torch.FloatTensor(np.zeros(opt.num_sim,)))
        # decoder parameters
        self.XYW = Parameter(torch.FloatTensor(k_dim, k_dim))
        self.W_vec = Parameter(torch.FloatTensor(1, k_dim))
        
        if opt.init == 'xavier_normal':
            xavier_normal_(self.U.data)
            xavier_normal_(self.V.data)
            
            xavier_normal_(self.att_weightX1.data)
            xavier_normal_(self.att_weightY1.data)
            xavier_normal_(self.att_weightX2.data)
            xavier_normal_(self.att_weightY2.data)
            
            xavier_normal_(self.w_w)
            xavier_normal_(self.h_w)
            xavier_normal_(self.h_t)   
            
            xavier_normal_(self.XYW.data)
            xavier_normal_(self.W_vec.data)
            
            xavier_normal_(self.att_vec_X)
            xavier_normal_(self.att_vec_Y)
        elif opt.init == 'xavier_uniform':
            xavier_uniform_(self.U.data)
            xavier_uniform_(self.V.data)
            
            xavier_uniform_(self.att_weightX1.data)
            xavier_uniform_(self.att_weightY1.data)
            xavier_uniform_(self.att_weightX2.data)
            xavier_uniform_(self.att_weightY2.data)
            
            xavier_uniform_(self.w_w)
            xavier_uniform_(self.h_w)
            xavier_uniform_(self.h_t)   
            
            xavier_uniform_(self.XYW.data)
            xavier_uniform_(self.W_vec.data)
            
            xavier_uniform_(self.att_vec_X)
            xavier_uniform_(self.att_vec_Y)
        elif opt.init == 'he_normal':
            kaiming_normal_(self.U.data)
            kaiming_normal_(self.V.data)            
           
            kaiming_normal_(self.att_weightX1.data)
            kaiming_normal_(self.att_weightY1.data)
            kaiming_normal_(self.att_weightX2.data)
            kaiming_normal_(self.att_weightY2.data)
            
            kaiming_normal_(self.w_w)
            kaiming_normal_(self.h_w)
            kaiming_normal_(self.h_t)   
            
            kaiming_normal_(self.XYW.data)
            kaiming_normal_(self.W_vec.data)    
            
            kaiming_normal_(self.att_vec_X)
            kaiming_normal_(self.att_vec_Y)            

        elif opt.init == 'he_uniform':
            kaiming_uniform_(self.U.data)
            kaiming_uniform_(self.V.data)
                        
            kaiming_uniform_(self.att_weightX1.data)
            kaiming_uniform_(self.att_weightY1.data)
            kaiming_uniform_(self.att_weightX2.data)
            kaiming_uniform_(self.att_weightY2.data)
            
            kaiming_uniform_(self.w_w)
            kaiming_uniform_(self.h_w)
            kaiming_uniform_(self.h_t)   
            
            kaiming_uniform_(self.XYW.data)
            kaiming_uniform_(self.W_vec.data)  
            
            kaiming_uniform_(self.att_vec_X)
            kaiming_uniform_(self.att_vec_Y)
    def get_final(self, E_ls, opt):
        E_final = 0
        opt.layer_att = 1
        if opt.layer_att:
            att_embed = F.softmax(self.att_embed, 0)
            # att_embed = self.att_embed / self.att_embed.sum()
            for i_layer in range(opt.num_layer):
                E_final = E_final + att_embed[i_layer] * E_ls[i_layer]
        return E_final
    def design_att(self, similarity_att, X, Y, idx0, idx1):
        if similarity_att == 9:
            X_embed5, Y_embed5 = self.sim_5(X, Y, idx0, idx1)
            X_embed8, Y_embed8 = self.sim_8(X, Y, idx0, idx1)
            X_embed = 0.5 * (X_embed5 + X_embed8)
            Y_embed = 0.5 * (Y_embed5 + Y_embed8)
        else:
            print('the similarity attention type is wrong')
            sys.exit(1)
        return X_embed, Y_embed
    def sim_5(self, X, Y, idx0, idx1):
            # cat linear transformation: A3NCF
            X_sample = X[:,idx0,:].transpose(0,1) # X_sample: sample, feat_num, feat_dim
            Y_sample = Y[:,idx1,:].transpose(0,1)
            
            X_cat = torch.cat([item for item in X_sample.transpose(0,1)],dim=1)
            Y_cat = torch.cat([item for item in Y_sample.transpose(0,1)],dim=1)
            
            att_X = F.softmax(self.att_cat_X2(F.leaky_relu(self.att_cat_X1(X_cat))), dim=-1)
            att_Y = F.softmax(self.att_cat_Y2(F.leaky_relu(self.att_cat_Y1(Y_cat))), dim=-1)
            
            X_embed = torch.mul(X_sample.sum(1), att_X)
            Y_embed = torch.mul(Y_sample.sum(1), att_Y)
            return X_embed, Y_embed
    def sim_8(self, X, Y, idx0, idx1):
        # 直接给予权重
        X_sample = X[:,idx0,:].transpose(0,1) # X_sample: sample, feat_num, feat_dim
        Y_sample = Y[:,idx1,:].transpose(0,1)
        
        X_att = F.softmax(self.x_att, dim=0)
        X_embed = torch.cat([(X_att[tt] * X_sample[:, tt, :]).unsqueeze(1) for tt in range(X_sample.shape[1])], dim=1).sum(1)
        
        Y_att = F.softmax(self.y_att, dim=0)
        Y_embed = torch.cat([(Y_att[tt] * Y_sample[:, tt, :]).unsqueeze(1) for tt in range(Y_sample.shape[1])], dim=1).sum(1)
        return X_embed, Y_embed
    def get_pred(self, opt, X_embed, Y_embed):
        if opt.score_type == 'none':
            pred = torch.mul(X_embed, Y_embed).sum(1)
        elif opt.score_type == 'ncf_linear':
            o = torch.mul(X_embed, Y_embed)
            o = F.leaky_relu(self.decoder1(o))
            pred = self.decoder2(o).flatten()
        elif opt.score_type == 'w':
            pred = torch.mul(torch.matmul(X_embed, self.XYW), Y_embed).sum(1)
        elif opt.score_type == 'vec':
            pred = torch.mul(X_embed, torch.mul(Y_embed, self.W_vec)).sum(1)
        elif (opt.score_type == 'cat'):
            feat = torch.cat((X_embed, Y_embed), dim=1)
            o = F.leaky_relu(self.decoder0(feat))
            o = F.leaky_relu(self.decoder1(o))
            self.tsneX = o.cpu().detach().numpy()
            pred = self.decoder2(o).flatten()
        if opt.sigmoid_flag:
            pred = nn.Sigmoid()(pred)
        return pred
    def get_feat_dgi_loss(self, F_u, F_i, feature_type, train_type):
        if feature_type == 'dgi':
            if train_type=='finetuning':
                _, feat_array = self.dgi.dgi_forward(self.association, np.eye(F_u.shape[0]+F_i.shape[0]))
                dgi_loss = torch.FloatTensor(np.array([0]))[0]
                feat_u = feat_array[: F_u.shape[0]]
                feat_i = feat_array[F_u.shape[0]: ]
            elif train_type=='one_stage':
                dgi_loss, feat_array = self.dgi.dgi_forward(self.association, np.eye(F_u.shape[0]+F_i.shape[0]))
                feat_u = feat_array[: F_u.shape[0]]
                feat_i = feat_array[F_u.shape[0]: ]
            elif train_type=='pretrain':
                dgi_loss = torch.FloatTensor(np.array([0]))[0]
                feat_u = F_u
                feat_i = F_i
            else:
                print('train_type '+ train_type +' is wrong')
                sys.exists(0)
        else:
            dgi_loss = torch.FloatTensor(np.array([0]))[0]
            feat_u = F_u
            feat_i = F_i
        return dgi_loss, feat_u, feat_i
    def forward(self, opt, F_u, F_i, idx0, idx1, adj_net_homo, adj_net_hete):
        ## F_u.shape: (user_sample_total_num, feat_dim)
        ## F_i.shape: (item_sample_total_num, feat_dim)
        ## idx0, idx1: batch_size, number of train or test
        ## adj_net_homo, adj_net_hete: shape: (user_sim_num * item_sim_num, user_total_num + item_total_num, user_total_num + item_total_num)
        
        dgi_loss, feat_u, feat_i = self.get_feat_dgi_loss(F_u, F_i, opt.feature_type, opt.train_type)

        
        X_u = t.matmul(feat_u, self.U)
        X_i = t.matmul(feat_i, self.V)
        E_0 = t.cat((X_u, X_i), 1)
        E_0 = F.dropout(E_0, p=self.dropout, training=self.training)
        
        E_ls = [E_0]
        for i_layer in range(opt.num_layer):
            E_ls.append(torch.cat([self.NIPLayer_ls[i_layer][j_sim](E_ls[i_layer][j_sim], adj_net_homo[j_sim], adj_net_hete[j_sim], self.row_num, opt.hh_fusion, opt.alp, opt.beta).unsqueeze(0) for j_sim in range(opt.num_sim)],0))
                
        ### 3 层 embedding 加权求和 或者 求平均
        E_final = self.get_final(E_ls, opt)
        
        X = E_final[:, :self.row_num, :]
        Y = E_final[:, self.row_num:, :]
        self.X_case = X.cpu().detach()
        self.Y_case = Y.cpu().detach()
        X_embed, Y_embed = self.design_att(opt.similarity_att, X, Y, idx0, idx1)
        
        pred = self.get_pred(opt, X_embed, Y_embed)

        return pred, dgi_loss