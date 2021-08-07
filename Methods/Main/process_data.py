# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 16:29:10 2020
Modified on Mon Jan 4 20点04分 2021
@author: xinxi
"""


import numpy as np
import pandas as pd
import os
import scipy.sparse as sp
import scipy.io as sio
import sys

from utils.similarity import get_Jaccard_Similarity as getSim
from utils.similarity import get_Gauss_Similarity
# from sklearn.metrics.pairwise import cosine_similarity

def read_data(dataName, readPrefix):
    def read_deepDR(dataPrefix):
        names = {'A': 'drug',
                 'b': 'disease'}
        # association matrix
        dataY = np.loadtxt(dataPrefix + 'drugDisease.txt', delimiter='\t', dtype=float)
        # drug
        drugdrugSim = np.loadtxt(dataPrefix + 'Sim_drugdrug.txt', delimiter='\t', dtype=float)
        drugProteinSim = np.loadtxt(dataPrefix + 'Sim_drugProtein.txt', delimiter='\t', dtype=float)
        drugsideEffectSim = np.loadtxt(dataPrefix + 'Sim_drugsideEffect.txt', delimiter='\t', dtype=float)
        drugsimBPnet = np.loadtxt(dataPrefix + 'drugsimBPnet.txt', delimiter='\t', dtype=float)
        drugsimCCnet = np.loadtxt(dataPrefix + 'drugsimCCnet.txt', delimiter='\t', dtype=float)
        drugsimChemicalnet = np.loadtxt(dataPrefix + 'drugsimChemicalnet.txt', delimiter='\t', dtype=float)
        drugsimMetanet = np.loadtxt(dataPrefix + 'drugsimMetanet.txt', delimiter='\t', dtype=float)
        drugsimMFnet = np.loadtxt(dataPrefix + 'drugsimMFnet.txt', delimiter='\t', dtype=float)
        drugsimTherapeuticnet = np.loadtxt(dataPrefix + 'drugsimTherapeuticnet.txt', delimiter='\t', dtype=float)
        drugsimWmnet = np.loadtxt(dataPrefix + 'drugsimWmnet.txt', delimiter='\t', dtype=float)
                
        AAr = np.array([drugdrugSim, drugProteinSim, drugsideEffectSim, drugsimBPnet, drugsimCCnet, drugsimChemicalnet, 
                        drugsimMetanet, drugsimMFnet, drugsimTherapeuticnet, drugsimWmnet])
        ANet = {}
        # disease
        bAr = np.array([np.eye(dataY.shape[1])])
        bNet = {}
        return dataY, AAr, bAr, ANet, bNet, names
    def read_LuoDTI(dataPrefix):
        names = {'A': 'drug',
                 'b': 'protein'}
        # association matrix
        dataY = np.loadtxt(dataPrefix + 'mat_drug_protein.txt', dtype=float, delimiter=' ')
        # drug
        drug_sim = pd.read_csv(dataPrefix + 'Similarity_Matrix_Drugs.txt',delimiter='    ', 
                               engine='python', header=None, dtype=float).values
    
        drug_dis = np.loadtxt(dataPrefix + 'mat_drug_disease.txt', dtype=float, delimiter=' ')
        drug_drug = np.loadtxt(dataPrefix + 'mat_drug_drug.txt', dtype=float, delimiter=' ')
        drug_se = np.loadtxt(dataPrefix + 'mat_drug_se.txt', dtype=float, delimiter=' ')
    
        drug_dis_sim = getSim(drug_dis).A
        drug_drug_sim = getSim(drug_drug).A
        drug_se_sim = getSim(drug_se).A
    
        AAr=np.array([drug_sim, drug_dis_sim, drug_drug_sim, drug_se_sim])
        ANet={'drug_dis': drug_dis,
              'drug_drug': drug_drug,
              'drug_se': drug_se}
        # protein
        protein_sim = np.loadtxt(dataPrefix + 'Similarity_Matrix_Proteins+trans.txt', dtype=float, delimiter='\t')
    
        protein_dis = np.loadtxt(dataPrefix + 'mat_protein_disease.txt', dtype=float, delimiter=' ')
        protein_protein = np.loadtxt(dataPrefix + 'mat_protein_protein.txt', dtype=float, delimiter=' ')
    
        protein_dis_sim = getSim(protein_dis).A
        protein_protein_sim = getSim(protein_protein).A
    
        bAr=np.array([protein_sim, protein_dis_sim, protein_protein_sim])
        bNet = {'protein_dis': protein_dis,
                'protein_protein': protein_protein}
        return dataY, AAr, bAr, ANet, bNet, names
    def read_Enzyme(dataPrefix):
        names = {'A': 'drug',
                 'b': 'target'}
        # association matrix
        dataY=np.loadtxt(dataPrefix+'e_admat_dgc_noRowCol1.txt',dtype=float,delimiter='\t').transpose()
        # drug
        dr_dr_sim=np.loadtxt(dataPrefix+'e_simmat_dc_noRowCol1.txt',dtype=float,delimiter='\t')
        AAr=np.array([dr_dr_sim])
        ANet = {}
        # target
        dis_sim = np.loadtxt(dataPrefix+'e_simmat_dg_noRowCol1.txt',dtype=float,delimiter='\t')
        bAr = np.array([dis_sim])
        bNet = {}
        return dataY, AAr, bAr, ANet, bNet, names
    def read_ZhangMDA(dataPrefix):
        names = {'A': 'miRNA',
                 'b': 'disease'}
        # association matrix
        dataY = np.loadtxt(dataPrefix + 'mi_dis.csv',delimiter=',',dtype=float)
        # miRNA
        mi_func_sim = np.loadtxt(dataPrefix + 'miRNA_LLS_similarity.csv', delimiter=',', dtype=float)
        AAr = np.array([mi_func_sim])
        ANet = {}
        # disease
        dis_sim = np.loadtxt(dataPrefix + 'mesh_re.csv', delimiter=',', dtype=float)
        bAr = np.array([dis_sim])
        bNet = {}
        return dataY, AAr, bAr, ANet, bNet, names
    def read_gene_disease(dataPrefix):
        names = {'A': 'gene',
                 'b': 'disease'}
        # association matrix
        dataY=np.loadtxt(dataPrefix+'gene-dis.csv',dtype=float,delimiter=',')
        # gene
        gene_gene_sim=np.loadtxt(dataPrefix+'Sim_gene-gene-network.txt',dtype=float,delimiter='\t')
        gene_chem_sim=np.loadtxt(dataPrefix+'Sim_gene-chem.txt',dtype=float,delimiter='\t')
        AAr=np.array([gene_gene_sim, gene_chem_sim])
        ANet = {}
        # disease
        dis_sim = np.loadtxt(dataPrefix+'Sim_dis-chem.txt',dtype=float,delimiter='\t')
        bAr = np.array([dis_sim])
        bNet = {}
        return dataY, AAr, bAr, ANet, bNet, names
    def read_GPCR(dataPrefix):
        names = {'A': 'drug',
                 'b': 'target'}
        # association matrix
        dataY=np.loadtxt(dataPrefix+'gpcr_admat_dgc_noRowCol1.txt',dtype=float,delimiter='\t').transpose()
        # drug
        dr_dr_sim=np.loadtxt(dataPrefix+'gpcr_simmat_dc_noRowCol1.txt',dtype=float,delimiter='\t')
        AAr=np.array([dr_dr_sim])
        ANet = {}
        # target
        dis_sim = np.loadtxt(dataPrefix+'gpcr_simmat_dg_noRowCol1.txt',dtype=float,delimiter='\t')
        bAr = np.array([dis_sim])
        bNet = {}
        return dataY, AAr, bAr, ANet, bNet, names
    def read_IC(dataPrefix):
        names = {'A': 'drug',
                 'b': 'target'}
        # association matrix
        dataY=np.loadtxt(dataPrefix+'ic_admat_dgc_noRowCol1.txt',dtype=float,delimiter='\t').transpose()
        # drug
        dr_dr_sim=np.loadtxt(dataPrefix+'ic_simmat_dc_noRowCol1.txt',dtype=float,delimiter='\t')
        AAr=np.array([dr_dr_sim])
        ANet = {}
        # target
        dis_sim = np.loadtxt(dataPrefix+'ic_simmat_dg_noRowCol1.txt',dtype=float,delimiter='\t')
        bAr = np.array([dis_sim])
        bNet = {}
        return dataY, AAr, bAr, ANet, bNet, names
    def read_LiuDTI(dataPrefix):
        names = {'A': 'drug',
                 'b': 'target'}
        # association matrix
        dataY=np.loadtxt(dataPrefix+'random_negative_interaction_matrix_noRowCol1.txt',dtype=float,delimiter='\t').transpose()
        # drug
        dr_dr_sim=np.loadtxt(dataPrefix+'chemical_simility_matrix_noRowCol1.txt',dtype=float,delimiter='\t')
        AAr=np.array([dr_dr_sim])
        ANet = {}
        # target
        dis_sim = np.loadtxt(dataPrefix+'protein_similarity_matrix_noRowCol1.txt',dtype=float,delimiter='\t')
        bAr = np.array([dis_sim])
        bNet = {}
        return dataY, AAr, bAr, ANet, bNet, names
    def read_ImpHuman(dataPrefix):
        names = {'A': 'drug',
                 'b': 'target'}
        # association matrix
        dataY=np.loadtxt(dataPrefix+'random_negative_interaction_matrix_noRowCol1.txt',dtype=float,delimiter='\t').transpose()
        # drug
        dr_dr_sim=np.loadtxt(dataPrefix+'chemical_simility_matrix_noRowCol1.txt',dtype=float,delimiter='\t')
        AAr=np.array([dr_dr_sim])
        ANet = {}
        # target
        dis_sim = np.loadtxt(dataPrefix+'protein_similarity_matrix_noRowCol1.txt',dtype=float,delimiter='\t')
        bAr = np.array([dis_sim])
        bNet = {}
        return dataY, AAr, bAr, ANet, bNet, names
    def read_Liu(dataPrefix):
        names = {'A': 'drug',
                 'b': 'side-effect'}
        data_dict = sio.loadmat(dataPrefix + 'Liu_dataset.mat')
        # association matrix        
        drug_se = np.array(data_dict['side_effect'],dtype=np.int)
        # drug
        drug_sub = data_dict['chemical']
        drug_target = data_dict['Targets']
        drug_transporter = data_dict['Transporters']
        drug_enzyme = data_dict['Enzymes']
        drug_pathway = data_dict['Pathways']
        drug_indication = data_dict['Treatment']
        
        drug_sub_sim = getSim(drug_sub).A
        drug_target_sim = getSim(drug_target).A
        drug_transporter_sim = getSim(drug_transporter).A
        drug_enzyme_sim = getSim(drug_enzyme).A
        drug_pathway_sim = getSim(drug_pathway).A
        drug_indication_sim = getSim(drug_indication).A
        
        AAr = np.array([drug_sub_sim, drug_target_sim, drug_transporter_sim, 
                        drug_enzyme_sim, drug_pathway_sim, drug_indication_sim])
        ANet = {}
        # side effect
        bAr = np.array([np.eye(drug_se.shape[1])])
        bNet = {}
        return drug_se, AAr, bAr, ANet, bNet, names
    def read_LiangDDA(dataPrefix):
        names = {'A': 'drug',
                 'b': 'disease'}
        # association matrix
        dataY=np.loadtxt(dataPrefix+'drug_dis_mat_noRowCol1.txt',dtype=float,delimiter='\t')
        # drug
        dr_pubchem_sim=np.loadtxt(dataPrefix+'network/Sim_drug_pubchem_mat_noRowCol1.txt',dtype=float,delimiter='\t')
        dr_domain_sim=np.loadtxt(dataPrefix+'network/Sim_drug_target_domain_mat_noRowCol1.txt',dtype=float,delimiter='\t')
        dr_go_sim=np.loadtxt(dataPrefix+'network/Sim_drug_target_go_mat_noRowCol1.txt',dtype=float,delimiter='\t')
    
        AAr=np.array([dr_pubchem_sim,dr_domain_sim,dr_go_sim])
        ANet = {}
        # disease
        dis_sim=np.loadtxt(dataPrefix+'network/Sim_mat_disease.txt',dtype=float,delimiter='\t')
        bAr=np.array([dis_sim])
        bNet = {}
        return dataY, AAr, bAr, ANet, bNet, names
    def read_NIMCD1(dataPrefix):
        names = {'A': 'miRNA',
                 'b': 'disease'}
        # association matrix
        dataY = np.loadtxt(dataPrefix + 'm-d.csv',delimiter=',',dtype=float)
        # miRNA
        mi_func_sim = np.loadtxt(dataPrefix + 'm-m.csv', delimiter=',', dtype=float)
        AAr = np.array([mi_func_sim])
        ANet = {}
        # disease
        dis_sim = np.loadtxt(dataPrefix + 'd-d.csv', delimiter=',', dtype=float)
        bAr = np.array([dis_sim])
        bNet = {}
        return dataY, AAr, bAr, ANet, bNet, names
    def read_LiMDA(dataPrefix):
        names = {'A': 'miRNA',
                 'b': 'disease'}
        # association matrix
        dataY = np.loadtxt(dataPrefix + 'm-d.txt',delimiter=',',dtype=float)
        # miRNA
        mi_func_sim = np.loadtxt(dataPrefix + 'm-m.txt', delimiter=',', dtype=float)
        AAr = np.array([mi_func_sim])
        ANet = {}
        # disease
        dis_sim = np.loadtxt(dataPrefix + 'd-d.txt', delimiter=',', dtype=float)
        bAr = np.array([dis_sim])
        bNet = {}
        return dataY, AAr, bAr, ANet, bNet, names
    def read_Nuclear(dataPrefix):
        names = {'A': 'drug',
                 'b': 'target'}
        # association matrix
        dataY=np.loadtxt(dataPrefix+'nr_admat_dgc_noRowCol1.txt',dtype=float,delimiter='\t').transpose()
        # drug
        dr_dr_sim=np.loadtxt(dataPrefix+'nr_simmat_dc_noRowCol1.txt',dtype=float,delimiter='\t')
        AAr=np.array([dr_dr_sim])
        ANet = {}
        # target
        dis_sim = np.loadtxt(dataPrefix+'nr_simmat_dg_noRowCol1.txt',dtype=float,delimiter='\t')
        bAr = np.array([dis_sim])
        bNet = {}
        return dataY, AAr, bAr, ANet, bNet, names
    def read_ZhangDDA(dataPrefix):
        names = {'A': 'drug',
                 'b': 'disease'}
        # association matrix
        dataY=np.loadtxt(dataPrefix+'dr_dis_association_mat.txt',dtype=float,delimiter=' ')
        # drug
        dr_enzyme_sim=np.loadtxt(dataPrefix+'enzyme_sim.txt',dtype=float,delimiter=' ')
        dr_target_sim=np.loadtxt(dataPrefix+'target_sim.txt',dtype=float,delimiter=' ')
        dr_struct_sim=np.loadtxt(dataPrefix+'structure_sim.txt',dtype=float,delimiter=' ')
        dr_pathwy_sim=np.loadtxt(dataPrefix+'pathway_sim.txt',dtype=float,delimiter=' ')
        dr_intera_sim=np.loadtxt(dataPrefix+'drug_interaction_sim.txt',dtype=float,delimiter=' ')
        AAr=np.array([dr_enzyme_sim,dr_target_sim,dr_struct_sim,dr_pathwy_sim,dr_intera_sim])
        ANet = {}
        # disease
        dis_sim=np.loadtxt(dataPrefix+'dis_sim.txt',dtype=float,delimiter=' ')
        bAr=np.array([dis_sim])
        bNet = {}
        return dataY, AAr, bAr, ANet, bNet, names
    def read_SFPEL(dataPrefix):
        names = {'A': 'LNCRNA',
                 'b': 'Protein'}
        data_dict = sio.loadmat(dataPrefix + 'NP2.0')
        # association matrix
        dataY = np.array(data_dict['InteractionMatrix'], dtype=float)
        # lnc RNA
        lnc_PCP_feat = data_dict['PCPseDNCFeature_LNCRNA']
        lnc_SCP_feat = data_dict['SCPseDNCFeature_LNCRNA']
        
        lnc_PCP_sim = get_Gauss_Similarity(lnc_PCP_feat).A
        lnc_SCP_sim = get_Gauss_Similarity(lnc_SCP_feat).A
        lnc_subgraph_sim = data_dict['lncRNA_subgraph_similarity_normalize']
        
        AAr = np.array([lnc_PCP_sim, lnc_SCP_sim, lnc_subgraph_sim])
        ANet = {}
        # protein
        protein_PCP_feat = data_dict['PCPseAACFeature_Protein']
        protein_SCP_feat = data_dict['SCPseAACFeature_Protein']
        
        protein_PCP_sim = get_Gauss_Similarity(protein_PCP_feat).A
        protein_SCP_sim = get_Gauss_Similarity(protein_SCP_feat).A
        protein_subgraph_sim = data_dict['protein_subgraph_similarity_normalize']
        
        bAr = np.array([protein_PCP_sim, protein_SCP_sim, protein_subgraph_sim])
        bNet = {}
        return dataY, AAr, bAr, ANet, bNet, names
    if dataName.startswith('deepDR'):
        print('process deepDR data')
        dataY, AAr, bAr, ANet, bNet, names = read_deepDR(readPrefix)
    elif dataName.startswith('LuoDTI'):
        print('process DTINet data')
        dataY, AAr, bAr, ANet, bNet, names = read_LuoDTI(readPrefix)
    elif dataName.startswith('Enzyme'):
        print('process Enzyme data')
        dataY, AAr, bAr, ANet, bNet, names = read_Enzyme(readPrefix)
    elif dataName.startswith('ZhangMDA'):
        print('process ZhangMDA data')
        dataY, AAr, bAr, ANet, bNet, names = read_ZhangMDA(readPrefix)
    elif dataName.startswith('gene_disease'):
        print('process gene_disease data')
        dataY, AAr, bAr, ANet, bNet, names = read_gene_disease(readPrefix)
    elif dataName.startswith('GPCR'):
        print('process GPCR data')
        dataY, AAr, bAr, ANet, bNet, names = read_GPCR(readPrefix)
    elif dataName.startswith('IC'):
        print('process IC data')
        dataY, AAr, bAr, ANet, bNet, names = read_IC(readPrefix)
    elif dataName.startswith('LiuDTI'):
        print('process LiuDTI data')
        dataY, AAr, bAr, ANet, bNet, names = read_LiuDTI(readPrefix)
    elif dataName.startswith('ImpHuman'):
        print('process ImpHuman data')
        dataY, AAr, bAr, ANet, bNet, names = read_ImpHuman(readPrefix)
    elif dataName.startswith('Liu'):
        print('process Liu data')
        dataY, AAr, bAr, ANet, bNet, names = read_Liu(readPrefix)    
    elif dataName.startswith('LiangDDA'):
        print('process LRSSL data')
        dataY, AAr, bAr, ANet, bNet, names = read_LiangDDA(readPrefix)
    elif dataName.startswith('NIMCD1'):
        print('process NIMCD1 data')
        dataY, AAr, bAr, ANet, bNet, names = read_NIMCD1(readPrefix)
    elif dataName.startswith('LiMDA'):
        print('process NIMCD2 data')
        dataY, AAr, bAr, ANet, bNet, names = read_LiMDA(readPrefix)
    elif dataName.startswith('Nuclear'):
        print('process Nuclear data')
        dataY, AAr, bAr, ANet, bNet, names = read_Nuclear(readPrefix)
    elif dataName.startswith('ZhangDDA'):
        print('process SCMFDD data')
        dataY, AAr, bAr, ANet, bNet, names = read_ZhangDDA(readPrefix)
    elif dataName.startswith('SFPEL'):
        print('process SFPEL data')
        dataY, AAr, bAr, ANet, bNet, names = read_SFPEL(readPrefix)    
    else:
        print('error: no data named '+dataName)
        sys.exit()
    return dataY, AAr, bAr, ANet, bNet, names


def splitData(dataY, outPrefix, nfold):
    # dataY为association矩阵，然后将其划分为k折的数据，分开存储
    # neg_pos_ratio: 表示负样本是正样本数量的几倍
    # tes_ratio：表示测试集所占的比例
    neg_pos_ratio = 1.0
    tes_ratio = 0.1
    seed = 1
    
    index_pos = np.array(np.where(dataY == 1))
    index_neg = np.array(np.where(dataY == 0))
    pos_num = len(index_pos[0])
    neg_num = int(pos_num * neg_pos_ratio)
    np.random.seed(seed)
    np.random.shuffle(index_pos.T)
    np.random.seed(seed)
    np.random.shuffle(index_neg.T)
    index_neg = index_neg[:, : neg_num]
    
    tes_pos_num = int(pos_num * tes_ratio)
    tes_neg_num = int(neg_num * tes_ratio)
    tes_index = np.hstack((index_pos[:, : tes_pos_num], index_neg[:, : tes_neg_num]))
    tes_data = np.hstack((tes_index.T, dataY[tes_index[0], tes_index[1]].reshape(-1, 1)))
    np.savetxt(outPrefix + 'tes_seed' + str(seed) + '.txt', tes_data, fmt='%d', delimiter=',')
    
    tes_data_total = tes_data[tes_data[:,-1]==1][:,:-1]
    tes_data_total[:,1]+=dataY.shape[0]
    np.savetxt(outPrefix + 'tes_seed' + str(seed) + '_total.txt', tes_data_total, fmt='%d', delimiter=' ')
    
    tra_val_pos = index_pos[:, tes_pos_num: ]
    tra_val_neg = index_neg[:, tes_neg_num: ]
    
    fold_index_pos = np.array([temp % nfold for temp in range(len(tra_val_pos[0]))])
    fold_index_neg = np.array([temp % nfold for temp in range(len(tra_val_neg[0]))])
    
    kfold=0
    for kfold in range(nfold):
        tra_fold_pos = tra_val_pos.T[fold_index_pos != kfold]
        val_fold_pos = tra_val_pos.T[fold_index_pos == kfold]
        tra_fold_neg = tra_val_neg.T[fold_index_neg != kfold]
        val_fold_neg = tra_val_neg.T[fold_index_neg == kfold]
        
        tra_fold = np.vstack((tra_fold_pos, tra_fold_neg))
        val_fold = np.vstack((val_fold_pos, val_fold_neg))
        
        tra_data = np.hstack((tra_fold, dataY[tra_fold[:, 0], tra_fold[:, 1]].reshape(-1, 1)))        
        val_data = np.hstack((val_fold, dataY[val_fold[:, 0], val_fold[:, 1]].reshape(-1, 1)))
        tra_matx = sp.coo_matrix((tra_data[:, 2], (tra_data[:, 0],tra_data[:, 1])), shape=(dataY.shape[0],dataY.shape[1])).toarray()
        val_matx = sp.coo_matrix((val_data[:, 2], (val_data[:, 0],val_data[:, 1])), shape=(dataY.shape[0],dataY.shape[1])).toarray()
        np.savetxt(outPrefix + 'tra_kfold' + str(kfold) + '_seed' + str(seed) + '.txt', tra_data, fmt='%d', delimiter=',')
        np.savetxt(outPrefix + 'val_kfold' + str(kfold) + '_seed' + str(seed) + '.txt', val_data, fmt='%d', delimiter=',')
        np.savetxt(outPrefix + 'tra_mat_kfold' + str(kfold) + '_seed' + str(seed) + '.txt', tra_matx, fmt='%d', delimiter=',')
        np.savetxt(outPrefix + 'val_mat_kfold' + str(kfold) + '_seed' + str(seed) + '.txt', val_matx, fmt='%d', delimiter=',')
        tra_data_total = tra_data[tra_data[:, -1]==1][:, :-1]
        val_data_total = val_data[val_data[:, -1]==1][:, :-1]
        tra_data_total[:,1]+=dataY.shape[0]
        val_data_total[:,1]+=dataY.shape[0]
        np.savetxt(outPrefix + 'tra_kfold' + str(kfold) + '_seed' + str(seed) + '_total.txt', tra_data_total, fmt='%d', delimiter=' ')
        np.savetxt(outPrefix + 'val_kfold' + str(kfold) + '_seed' + str(seed) + '_total.txt', val_data_total, fmt='%d', delimiter=' ')
    temp = np.hstack((tra_val_pos, tra_val_neg))
    tra_val_data = np.hstack((temp.T, dataY[temp[0], temp[1]].reshape(-1, 1)))
    tra_val_matx = sp.coo_matrix((tra_val_data[:, 2], (tra_val_data[:, 0],tra_val_data[:, 1])), shape=(dataY.shape[0],dataY.shape[1])).toarray()
    tra_val_data_total = tra_val_data[tra_val_data[:, -1]==1][:, :-1]
    tra_val_data_total[:,1]+=dataY.shape[0]
    np.savetxt(outPrefix + 'tra_val_seed' + str(seed) + '.txt', tra_val_data, fmt='%d', delimiter=',')
    np.savetxt(outPrefix + 'tra_val_mat_seed' + str(seed) + '.txt', tra_val_matx, fmt='%d', delimiter=',')
    np.savetxt(outPrefix + 'tra_val_seed' + str(seed) + '_total.txt', tra_val_data_total, fmt='%d', delimiter=' ')
    
    np.savetxt(outPrefix + 'tra_kfold' + str(-1) + '_seed' + str(seed) + '.txt', tra_val_data, fmt='%d', delimiter=',')
    np.savetxt(outPrefix + 'tra_mat_kfold'+ str(-1) + '_seed' + str(seed) + '.txt', tra_val_matx, fmt='%d', delimiter=',')
    np.savetxt(outPrefix + 'tra_kfold' + str(-1) + '_seed' + str(seed) + '_total.txt', tra_val_data_total, fmt='%d', delimiter=' ')
    return

def splitDataMain(nfold, dataName):
    dataPath = '../../Datasets/'+dataName + '/'
    outPrefix = dataPath + 'split_data_tra_val_'+str(nfold)+'nfold/'
    if not os.path.exists(outPrefix):
        os.mkdir(outPrefix)
    dataPrefix = dataPath+'used_data/'
    # 读取association数据
    dataY, AAr, bAr, ANet, bNet, names = read_data(dataName, dataPrefix)
    # 划分数据为k折
    splitData(dataY, outPrefix, nfold)
    return dataY, AAr, bAr, ANet, bNet, names
if __name__ == '__main__':
    '''
    运行的命令行：python process_data_main.py fold_number dataName
    example: python process_data_main.py 5 DTINet_data
    example: python process_data_main.py 10 DTINet_data
    '''
    argv = sys.argv
    nfold = int(argv[1])
    dataName = argv[2]
    print(argv)
    dataY, AAr, bAr, ANet, bNet, names = splitDataMain(nfold, dataName)
    total_number = dataY.sum()
    sparsity = 1 - total_number / float(dataY.shape[0] * dataY.shape[1])
    print(dataY.shape, total_number, round(sparsity * 100,2))




