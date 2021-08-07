# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 00:23:35 2021

@author: xinxi
"""

import numpy as np
import sys
import torch

def caseFun(opt, model, tra_list):
    if opt.case_x > -1:
        if ((tra_list[:,0]==opt.case_x)&(tra_list[:,2]==1)).sum() < 1:
            sys.exit(1)
        if ((tra_list[:,0]==opt.case_x)&(tra_list[:,2]==0)).sum() < 1:
            sys.exit(1)
    elif opt.case_y > -1:
        if ((tra_list[:,1]==opt.case_y)&(tra_list[:,2]==1)).sum() < 1:
            sys.exit(1)
        if ((tra_list[:,1]==opt.case_y)&(tra_list[:,2]==0)).sum() < 1:
            sys.exit(1)
    else:
        print('index is wrong')
        sys.exit(1)
    
    case_x = opt.case_x
    case_y = opt.case_y
    if (case_x == -1) & (case_y != -1):
        idx0 = torch.LongTensor(range(opt.row_num))
        idx1 = torch.LongTensor([case_y] * opt.row_num)
    elif (case_x != -1) & (case_y == -1):
        idx0 = torch.LongTensor([case_x] * opt.col_num)
        idx1 = torch.LongTensor(range(opt.col_num))
    else:
        print('index is wrong')
        sys.exit(1)
        
    X_embed, Y_embed = model.design_att(opt.similarity_att, model.X_case, model.Y_case, idx0, idx1)
    pred = model.get_pred(opt, X_embed, Y_embed).cpu().detach()
    trip_result = np.vstack((idx0.numpy(), idx1.numpy(), pred.numpy())).transpose()
    
    fname = opt.dataName + '/' + '_'.join(['case study', 'PyIndX', str(case_x), 'PyIndY', str(case_y)]) + '.csv'
    np.savetxt(fname, trip_result, fmt='%.5f', delimiter=',')
    return
    
    
    