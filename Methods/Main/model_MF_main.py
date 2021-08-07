#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 02:32:28 2021

@author: fht
"""


import copy
import numpy as np
import scipy.sparse as sp
import sys
import time
import torch
import torch.optim as optim

from sklearn.metrics import roc_auc_score, average_precision_score

from model_MF import lossF, MF
from process_feature import getFeature
from torch_data import getLoader

from utils.clac_metric import get_metrics
from utils.normalization import normalizeRow, normalizeRowCol
from utils.process_set import diagZeroAdj,get_profile_sim


def getBigAdj(sim_A_total, sim_b_total, traiY):
    adj_net_homo = np.array([np.vstack((np.hstack((sim_A_total[i], np.zeros_like(traiY))),
                                    np.hstack((np.zeros_like(traiY.T), sim_b_total[j]))
                          )) for i in range(len(sim_A_total)) for j in range(len(sim_b_total))]
                        )
    adj_net_hete = np.array([np.vstack((np.hstack((np.zeros_like(sim_A_total[i]), traiY)),
                               np.hstack((traiY.T, np.zeros_like(sim_b_total[j])))
                              )) for i in range(len(sim_A_total)) for j in range(len(sim_b_total))]
                            )
    
    for i in range(len(adj_net_homo)):
        adj_net_homo[i] = normalizeRowCol(adj_net_homo[i])
        adj_net_hete[i] = normalizeRowCol(adj_net_hete[i])
    
    return torch.FloatTensor(adj_net_homo), torch.FloatTensor(adj_net_hete)



def testModel(opt, model, loader, F_u, F_i, adj_net_homo, adj_net_hete):
    model.eval()
    y_label, y_pred = [], []
    with torch.no_grad():
        for i, (idx0, idx1, y) in enumerate(loader):
            if opt.cuda:
                y = y.cuda()
            output, dgi_loss = model(opt, F_u, F_i, idx0, idx1, adj_net_homo, adj_net_hete)
            loss = opt.dgi_weight * dgi_loss.item() + lossF(opt, output, y).item()
            if opt.prin:
                if i == 0:
                    print('test iteration {:04d} / test loss: {:.4f} / test dgi loss {:.4f}'.format(i, loss, dgi_loss.item()))
            
            y_label += y.cpu().numpy().tolist()
            y_pred += output.cpu().numpy().tolist()
    return y_label, y_pred, average_precision_score(y_label, y_pred), roc_auc_score(y_label, y_pred), loss


def trainModel(opt, model, optimizer, adj_net_homo, adj_net_hete, F_u, F_i, trai_loader, vali_loader, test_loader, tes_list):
    max_auc = 0
    if opt.cuda:
        model.cuda()
        adj_net_homo.cuda()
        adj_net_hete.cuda()
        F_u.cuda()
        F_i.cuda()
    # Train model
    t_total = time.time()    
    if opt.prin:
        print('Start Training...')
    model_max = copy.deepcopy(model)
    loss_history = []
    epoch_patience = 0
    epoch = 0
    epoch_best = 0
    try:
        for epoch in range(opt.epochs):
            epoch_start = time.time()
            if opt.prin:
                print('\n-------- Epoch {:04d} --------'.format(epoch))
            y_pred_train = []
            y_label_train = []
            
            
            # train epoch
            model.train()
            for i, (idx0, idx1, y) in enumerate(trai_loader):
                if opt.cuda:
                    idx0 = idx0.cuda()
                    idx1 = idx1.cuda()
                    y = y.cuda()
                optimizer.zero_grad()
                output, dgi_loss = model(opt, F_u, F_i, idx0, idx1, adj_net_homo, adj_net_hete)
                loss_train = lossF(opt, output, y)
                
                loss_train = opt.dgi_weight * dgi_loss + loss_train
                
                
                
                loss_history.append(loss_train.item())
                loss_train.backward()                
                optimizer.step()
                y_label_train += y.cpu().numpy().tolist()
                y_pred_train += output.cpu().detach().numpy().tolist()
                if opt.prin:
                    if i == 0:
                        print('epoch: {:04d}'.format(epoch), 
                              '/ train dgi loss: {:.4f}'.format(dgi_loss.item()),
                              '/ train iteration: {:04d}'.format(i),
                              '/ train loss: {:.4f}'.format(loss_train.item()))
                        # print(model.alpha_num[:5])
            prc_train = average_precision_score(y_label_train, y_pred_train)
            roc_train = roc_auc_score(y_label_train, y_pred_train)
            
            epoch_patience += 1
            if epoch_patience > opt.patience:
                print('best epoch is: {:04d}'.format(epoch_best))
                break
            temp_model = copy.deepcopy(model)
            
            
            # validation after each epoch
            if not opt.fastmode:
                model.eval()
                y_label_val, y_pred_val, prc_val, roc_val, loss_val = testModel(opt, model, vali_loader, F_u, F_i, adj_net_homo, adj_net_hete)
                if roc_val > max_auc:
                    epoch_best = epoch
                    epoch_patience = 0
                    model_max = copy.deepcopy(temp_model)
                    y_label_opt, y_pred_opt, prc_opt, auroc_opt, loss_opt = y_label_val, y_pred_val, prc_val, roc_val, loss_val
                    # torch.save(model, 'my_model.pkl')
                    max_auc = roc_val
                    
                    if opt.prin:
                        # print(model.alpha_num)
                        # print(model.alpha_l2)
                        print('epoch: {:04d}\n'.format(epoch),
                              ' loss_train:  {:.4f}\n'.format(loss_train.item()),
                              ' auprc_train: {:.4f}\n'.format(prc_train),
                              ' auroc_train: {:.4f}\n'.format(roc_train),
                              ' loss_val:    {:.4f}\n'.format(loss_val),
                              ' auprc_val:   {:.4f}\n'.format(prc_val),
                              ' auroc_val:   {:.4f}\n'.format(roc_val),
                              ' max_auroc:   {:.4f}'.format(max_auc))
            else:
                model_max = copy.deepcopy(model)
            if opt.prin:
                print('the {:04d} epoch take {:.4f} seconds'.format(epoch, time.time()-epoch_start))
    except KeyboardInterrupt:
        print('\nbest epoch is: {:04d}\ncurrent epoch is: {:04d}'.format(epoch_best, epoch))
        pass
    # plt.plot(loss_history)
    if opt.prin:
        print("\nOptimization Finished!")
        print("Total time elapsed: {:.2f}s".format(time.time() - t_total))
    
    # Testing
    model_max.eval()
    # test_label, test_score, prc_test, auroc_test, loss_test = testModel(opt, model_max, test_loader, F_u, F_i, adj_net_homo, adj_net_hete)
    test_label, test_score, prc_test, auroc_test, loss_test = y_label_opt, y_pred_opt, prc_opt, auroc_opt, loss_opt
    if opt.prin:
        print('loss_test: {:.4f}\n'.format(loss_test),
              ' auprc_test: {:.4f}\n'.format(prc_test), 
              ' auroc_test: {:.4f}'.format(auroc_test))

    return test_label, test_score, model_max


def MF_HF_main(opt, sim_A, sim_b, tra_list, val_list, tes_list):
    
    opt.att_dim = opt.k_dim
    
    trai_loader = getLoader(opt.batch_size, tra_list)
    vali_loader = getLoader(opt.batch_size, val_list)
    test_loader = getLoader(opt.batch_size, tes_list)
    
    sim_A_total, sim_b_total = copy.deepcopy(sim_A), copy.deepcopy(sim_b)
    sim_A_total = get_profile_sim(sim_A_total, tra_list, opt.row_num, opt.col_num)
    tra_T = np.zeros_like(tra_list)
    tra_T[:,0] = tra_list[:,1]
    tra_T[:,1] = tra_list[:,0]
    tra_T[:,2] = tra_list[:,2]
    sim_b_total = get_profile_sim(sim_b_total, tra_T, opt.col_num, opt.row_num)
    
    sim_A_total = diagZeroAdj(sim_A_total)
    sim_b_total = diagZeroAdj(sim_b_total)
    traiY = sp.coo_matrix((tra_list[:,2], (tra_list[:,0],tra_list[:,1])),
                          shape=(opt.row_num, opt.col_num), dtype=np.float32).toarray()
    
    adj_net_homo, adj_net_hete = getBigAdj(sim_A_total, sim_b_total, traiY)
    
    
    F_u, F_i = getFeature(opt, traiY, sim_A_total, sim_b_total)
    F_u, F_i = torch.FloatTensor(F_u), torch.FloatTensor(F_i)
    
    if opt.ind_sim==-1:
        print('all similarity')
        opt.num_sim = len(sim_A_total) * len(sim_b_total)
        model = MF(opt, traiY)
        optimizer = optim.Adam(model.parameters(),
                       lr=opt.lr, weight_decay=opt.weight_decay)
        test_label, test_score, model_max = trainModel(opt, model, optimizer, adj_net_homo, adj_net_hete, F_u, F_i, trai_loader, vali_loader, test_loader, tes_list)
    else:
        print('single similarity')
        if opt.ind_sim>len(adj_net_homo):
            print('the similarity No. is greater than network No.')
            sys.exit(2)
        opt.num_sim = 1
        model = MF(opt, traiY)
        optimizer = optim.Adam(model.parameters(),
                       lr=opt.lr, weight_decay=opt.weight_decay)
        test_label, test_score, model_max = trainModel(opt, model, optimizer, adj_net_homo[opt.ind_sim].unsqueeze(0), 
                                                       adj_net_hete[opt.ind_sim].unsqueeze(0), 
                                                       F_u, F_i, trai_loader, vali_loader, test_loader, tes_list)
    criteria_result = get_metrics(np.mat(test_label), np.mat(test_score))
    return test_label, test_score, criteria_result, model_max, F_u, F_i