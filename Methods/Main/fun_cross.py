#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 03:10:07 2021

@author: fht
"""


import numpy as np
from model_MF_main import MF_HF_main



def crossValidate(opt, sim_A, sim_b, tra_list_cross, tes_list_cross):
    fold_result = np.zeros((opt.nfold+2,7))
    if opt.single_fold:
        test_label, test_score, criteria_result, model_max, F_u, F_i = MF_HF_main(opt, sim_A, sim_b, tra_list_cross[opt.kfold], tes_list_cross[opt.kfold], tes_list_cross[opt.kfold])
        fold_result[opt.kfold] = np.array(criteria_result)
    else:
        for kfold in range(opt.nfold):
            opt.kfold = kfold
            test_label, test_score, criteria_result, model_max, F_u, F_i = MF_HF_main(opt, sim_A, sim_b, tra_list_cross[opt.kfold], tes_list_cross[opt.kfold], tes_list_cross[opt.kfold])
            fold_result[kfold] = np.array(criteria_result)
    fold_result[-2] = fold_result[:-2].mean(0)
    fold_result[-1] = np.std(fold_result[:-2],axis=0,ddof=0)
    with open(opt.resultTxt, 'a') as fobj:
        fobj.write('\n\n')
        for kfold in range(opt.nfold):
            fobj.write('fold  '+str(kfold)+',')
            [fobj.write(str(round(temp,4))+',') for temp in fold_result[kfold]]
            [fobj.write(key+','+str(value)+',') for key, value in vars(opt).items()]
            fobj.write('\n')
        
        fobj.write('average' + ',')
        [fobj.write(str(round(temp,4))+',') for temp in fold_result[-2]]
        [fobj.write(key+','+str(value)+',') for key, value in vars(opt).items()]
        fobj.write('\n')
        
        fobj.write('std    ' + ',')
        [fobj.write(str(round(temp,4))+',') for temp in fold_result[-1]]
        [fobj.write(key+','+str(value)+',') for key, value in vars(opt).items()]
        fobj.write('\n')
    return fold_result

