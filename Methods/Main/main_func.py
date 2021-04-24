#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 02:35:54 2021

@author: fht
"""

import numpy as np
import os
import torch

from process_para import optPara
from fun_pre import prepare
from write_pred import write_score

def call_main(opt):
    if opt.device.startswith('cuda') & (torch.cuda.is_available()):
        opt.cuda = True
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
    else:
        opt.cuda = False
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    # 结果文件参数
    opt.resultTxt = opt.dataName + '/' + opt.dataName + '_' + opt.feature_type + '_' + opt.train_type + '_' + 'resultTxt_' + opt.lossType + '_' + opt.result_key + '.csv'
    if not os.path.exists(opt.dataName + '/'):
        os.mkdir(opt.dataName + '/')
    sim_A, sim_b, tra_list_cross, val_list_cross, tra_list_test, tes_list_test = prepare(opt)
    if opt.cross:
        from fun_cross import crossValidate
        opt.kfold = opt.tra_fold
        cross_result = crossValidate(opt, sim_A, sim_b, tra_list_cross, val_list_cross)
    if opt.test:
        from fun_test import testing
        opt.kfold=-1
        test_label, test_score, criteria_result, model_max, F_u, F_i = testing(opt, sim_A, sim_b, tra_list_test, tes_list_test)
        write_score(test_label, test_score, opt.dataName)
        if opt.caseStudy:
            from case_study import caseFun
            caseFun(opt, model_max, tra_list_test)
        if opt.tsne_flag:
            from plot_get_feature import get_tsne
            get_tsne(opt, model_max, tes_list_test, F_u, F_i)
if __name__ == '__main__':
    opt = optPara()
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    call_main(opt)