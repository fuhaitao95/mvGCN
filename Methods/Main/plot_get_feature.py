# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 08:44:25 2021

@author: xinxi
"""

import numpy as np
import torch

def get_tsne(opt, model, tes_list_k, F_u, F_i):
    idx0 = tes_list_k[:, 0]
    idx1 = tes_list_k[:, 1]
    label = tes_list_k[:, 2]
    X_embed, Y_embed = model.design_att(opt.similarity_att, model.X_case, model.Y_case, torch.LongTensor(idx0), torch.LongTensor(idx1))
    pred = model.get_pred(opt, X_embed, Y_embed).cpu().detach()
    tsneX = model.tsneX
    
    F_u_ls = F_u.cpu().numpy().tolist()
    F_x_1 = np.array([F_u_ls[int(temp)] for temp in idx0])
    
    F_i_ls = F_i.cpu().numpy().tolist()
    F_x_2 = np.array([F_i_ls[int(temp)] for temp in idx1])
    
    
    rawX = np.hstack((F_x_1,F_x_2))
    
    fname = opt.dataName + '/' + '_'.join(['tsne', 'label']) + '.csv'
    np.savetxt(fname, label, fmt='%.1f', delimiter=',')
    
    fname = opt.dataName + '/' + '_'.join(['tsne', 'EmbeddingLastHidden', 'raw']) + '.csv'
    np.savetxt(fname, tsneX, fmt='%.6f', delimiter=',')
    plt_tsne(opt, 'EmbeddingLastHidden', tsneX, label)
    
    fname = opt.dataName + '/' + '_'.join(['tsne', 'EmbeddingInitial', 'raw']) + '.csv'
    np.savetxt(fname, rawX, fmt='%.6f', delimiter=',')
    plt_tsne(opt, 'EmbeddingInitial', rawX, label)
    
    return
def plt_tsne(opt, fig_marker, X, label):
    import matplotlib.pyplot as plt
    from sklearn import manifold
    import pandas as pd
    np.random.seed(1)
    pos_label = label==1
    neg_label = label==0
    plt.rcParams['font.sans-serif'] = ['Arial']
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=1)
    X_tsne = tsne.fit_transform(X)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    np.savetxt(opt.dataName + '/' + fig_marker+'_2d.csv', np.hstack((X_norm,label.reshape(-1,1))), fmt='%.4f', delimiter=',')
    
    plt.figure(figsize=(10, 10))
    plt.tick_params(labelsize=30)
    plt.scatter(X_norm[:,0][pos_label],X_norm[:,1][pos_label],edgecolors=('black'),color='r', label='pos')
    plt.scatter(X_norm[:,0][neg_label],X_norm[:,1][neg_label],edgecolors=('black'),color='g', label='neg')
    legend_font = {'family': 'Arial', 'style': 'normal','size': 20,  'weight': "bold"}
    plt.legend(loc='lower right', prop=legend_font)
    plt.savefig(opt.dataName + '/' + fig_marker+'.tif',dpi=350)
    return
    