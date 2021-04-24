#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 10:53:22 2021

@author: fht
"""


import numpy as np
def write_score(test_label, test_score, dataName):
    test_label, test_score = np.array(test_label), np.array(test_score)
    result = np.vstack((test_label, test_score)).T
    np.savetxt(dataName + '/' + dataName + '_test_label_score.csv', result, fmt='%f', delimiter=',')
    return