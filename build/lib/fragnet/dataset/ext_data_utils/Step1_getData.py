"""
This code was copied from https://github.com/jianglikun/DeepTTC/blob/main/Step1_getData.py
and modified
"""

import sys
import csv
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

class GetData():
    def __init__(self, data_dir):
        PATH = data_dir

        rnafile = PATH + '/Cell_line_RMA_proc_basalExp.txt'
        smilefile = PATH + '/smile_inchi.csv'
        pairfile = PATH + '/GDSC2_fitted_dose_response_25Feb20.xlsx'
        drug_infofile = PATH + "/Drug_listTue_Aug10_2021.csv"
        drug_thred = PATH + '/IC50_thred.txt'


        self.pairfile = pairfile
        self.drugfile = drug_infofile
        self.rnafile = rnafile
        self.smilefile = smilefile
        self.drug_thred = drug_thred

    def getDrug(self):
        drugdata = pd.read_csv(self.smilefile,index_col=0)
        return drugdata

    def _filter_pair(self,drug_cell_df):

        # ['DATA.908134', 'DATA.1789883', 'DATA.908120', 'DATA.908442'] not in index
        # not_index = ['908134', '1789883', '908120', '908442']
        not_index = [908134, 1789883, 908120, 908442]
        print(drug_cell_df.shape)
        drug_cell_df = drug_cell_df[~drug_cell_df['COSMIC_ID'].isin(not_index)]
        print(drug_cell_df.shape)


        pub_df = pd.read_csv(self.drugfile)
        pub_df = pub_df.dropna(subset=['PubCHEM'])
        pub_df = pub_df[(pub_df['PubCHEM'] != 'none') & (pub_df['PubCHEM'] != 'several')]
        print(drug_cell_df.shape)
        drug_cell_df = drug_cell_df[drug_cell_df['DRUG_ID'].isin(pub_df['drug_id'])]
        print(drug_cell_df.shape)
        return drug_cell_df

    def _stat_cancer(self,drug_cell_df):
    
        cancer_num = drug_cell_df['TCGA_DESC'].value_counts().shape[0]
        min_cancer_drug = min(drug_cell_df['TCGA_DESC'].value_counts())
        max_cancer_drug = max(drug_cell_df['TCGA_DESC'].value_counts())
        mean_cancer_drug = np.mean(drug_cell_df['TCGA_DESC'].value_counts())


    def _stat_cell(self, drug_cell_df):

        cell_num = drug_cell_df['COSMIC_ID'].value_counts().shape[0]

        min_drug = min(drug_cell_df['COSMIC_ID'].value_counts())
        max_drug = max(drug_cell_df['COSMIC_ID'].value_counts())
        mean_drug = np.mean(drug_cell_df['COSMIC_ID'].value_counts())

    def _stat_drug(self, drug_cell_df):

        drug_num = drug_cell_df['DRUG_ID'].value_counts().shape[0]

        min_cell = min(drug_cell_df['DRUG_ID'].value_counts())
        max_cell = max(drug_cell_df['DRUG_ID'].value_counts())
        mean_cell = np.mean(drug_cell_df['DRUG_ID'].value_counts())


    def _split(self,df,col,ratio,random_seed):

        col_list = df[col].value_counts().index
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()

        for instatnce in col_list:
            sub_df = df[df[col] == instatnce]
            sub_df = sub_df[['DRUG_ID', 'COSMIC_ID','TCGA_DESC', 'LN_IC50', 'AUC']]
            sub_train, sub_test = train_test_split(sub_df, test_size=ratio,random_state=random_seed)
            if train_data.shape[0] == 0:
                train_data = sub_train
                test_data = sub_test
            else:
                # train_data = train_data.append(sub_train)
                # test_data = test_data.append(sub_test)

                train_data = pd.concat([train_data, sub_train], axis=0, ignore_index=True)
                test_data = pd.concat([test_data, sub_test], axis=0, ignore_index=True)
        


        return train_data,test_data

    def ByCancer(self,random_seed, test_size=.05):


        drug_cell_df = pd.read_excel(self.pairfile)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)
        drug_cell_df = self._filter_pair(drug_cell_df)

        # drug_cell_df = drug_cell_df.head(10000)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)

        print(drug_cell_df['TCGA_DESC'].value_counts())

        train_data, test_data = self._split(df=drug_cell_df, col='TCGA_DESC',
                                            ratio=test_size,random_seed=random_seed)

        return train_data, test_data

    def ByDrug(self):
        drug_cell_df = pd.read_excel(self.pairfile)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)
        drug_cell_df = self._filter_pair(drug_cell_df)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)

        train_data,test_data = self._split(df=drug_cell_df,col='DRUG_ID',ratio=0.2)

        return train_data,test_data

    def ByCell(self):
        drug_cell_df = pd.read_excel(self.pairfile)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)
        drug_cell_df = self._filter_pair(drug_cell_df)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)

        train_data, test_data = self._split(df=drug_cell_df, col='COSMIC_ID', ratio=0.2)

        return train_data, test_data

    def MissingData(self):
        drug_cell_df = pd.read_excel(self.pairfile)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)
        drug_cell_df = self._filter_pair(drug_cell_df)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)

        cell_list = drug_cell_df['COSMIC_ID'].value_counts().index
        drug_list = drug_cell_df['DRUG_ID'].value_counts().index

        all_df = pd.DataFrame()
        dup_drug = []
        [dup_drug.extend([i]*len(cell_list)) for i in drug_list]
        all_df['DRUG_ID'] = dup_drug

        dup_cell = []
        for i in range(len(drug_list)):
            dup_cell.extend(cell_list)
        all_df['COSMIC_ID'] = dup_cell

        all_df['ID'] = all_df['DRUG_ID'].astype(str).str.cat(all_df['COSMIC_ID'].astype(str),sep='_')
        drug_cell_df['ID'] = drug_cell_df['DRUG_ID'].astype(str).str.cat(drug_cell_df['COSMIC_ID'].astype(str),sep='_')
        MissingData = all_df[~all_df['ID'].isin(drug_cell_df['ID'])]


        return drug_cell_df,MissingData

    def _LeaveOut(self,df,col,ratio=0.8,random_num=1):
        random.seed(random_num)
        col_list = list(set(df[col]))
        col_list = list(col_list)

        sub_start = int(len(col_list)/5)*random_num
        if random_num==4:
            sub_end = len(col_list)
        else:
            sub_end = int(len(col_list)/5)*(random_num+1)

        leave_instatnce = list(set(col_list)- set(col_list[sub_start:sub_end]))

        df = df[['DRUG_ID', 'COSMIC_ID', 'TCGA_DESC', 'LN_IC50']]
        train_data = df[df[col].isin(leave_instatnce)]
        test_data = df[~df[col].isin(leave_instatnce)]

        print(len(col_list))
        print(len(set(list(train_data[col]))))
        print(len(set(list(test_data[col]))))


        return train_data,test_data

    def Cell_LeaveOut(self,random):
        drug_cell_df = pd.read_excel(self.pairfile)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)
        drug_cell_df = self._filter_pair(drug_cell_df)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)

        traindata,testdata = self._LeaveOut(df=drug_cell_df,col='COSMIC_ID',ratio=0.8,random_num=random)

        return traindata,testdata

    def Drug_LeaveOut(self,random):
        drug_cell_df = pd.read_excel(self.pairfile)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)
        drug_cell_df = self._filter_pair(drug_cell_df)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)

        traindata, testdata = self._LeaveOut(df=drug_cell_df, col='DRUG_ID', ratio=0.8,random_num=random)

        return traindata, testdata
    

    def _split_no_balance_binary(self,df,col,ratio,random_seed):

        col_list = df[col].value_counts().index
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()

        for instatnce in col_list:
            sub_df = df[df[col] == instatnce]
            sub_df = sub_df[['DRUG_ID', 'COSMIC_ID','TCGA_DESC', 'LN_IC50','Binary_IC50']]
            sub_train, sub_test = train_test_split(sub_df, test_size=ratio,
                                                   random_state=random_seed)
            if train_data.shape[0] == 0:
                train_data = sub_train
                test_data = sub_test
            else:
                train_data = train_data.append(sub_train)
                test_data = test_data.append(sub_test)


        return train_data,test_data

    def _split_balance_binary(self,df,col,ratio,random_seed):

        col_list = df[col].value_counts().index

        pos_data = df[df[col]==1]
        neg_data = df[df[col]==0]

        down_pos_data = pos_data.loc[random.sample(list(pos_data.index),neg_data.shape[0])]

        combine_data = neg_data.append(down_pos_data)


        combine_data = combine_data[['DRUG_ID', 'COSMIC_ID','TCGA_DESC', 'LN_IC50','Binary_IC50']]

        train_data, test_data = train_test_split(combine_data, test_size=ratio,
                                                   random_state=random_seed)


        return train_data,test_data



    def getRna(self,traindata,testdata, use_genes=None):
        train_rnaid = list(traindata['COSMIC_ID'])
        test_rnaid = list(testdata['COSMIC_ID'])
        train_rnaid = ['DATA.'+str(i) for i in train_rnaid]
        test_rnaid = ['DATA.' +str(i) for i in test_rnaid ]


        # TODO: filter genes here
        # train_rnaid = ['GENE_SYMBOLS'] + train_rnaid
        # test_rnaid = ['GENE_SYMBOLS'] + test_rnaid
        
        rnadata =  pd.read_csv(self.rnafile,sep='\t')        
            
        train_rnadata = rnadata[train_rnaid]
        test_rnadata = rnadata[test_rnaid]

        return train_rnadata,test_rnadata



