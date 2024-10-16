import numpy as np
import pandas as pd
from .Step1_getData import GetData

class DataEncoding:
    def __init__(self, data_dir):
        self.Getdata = GetData(data_dir)

    def encode2(self,traindata, valdata, testdata):
        drug_smiles = self.Getdata.getDrug()
        drugid2smile = dict(zip(drug_smiles['drug_id'],drug_smiles['smiles']))

        traindata['smiles'] = [drugid2smile[i] for i in traindata['DRUG_ID']]
        valdata['smiles'] = [drugid2smile[i] for i in valdata['DRUG_ID']]
        testdata['smiles'] = [drugid2smile[i] for i in testdata['DRUG_ID']]


        traindata = traindata.reset_index()
        traindata['Label'] = traindata['LN_IC50']

        valdata = valdata.reset_index()
        valdata['Label'] = valdata['LN_IC50']


        testdata = testdata.reset_index()
        testdata['Label'] = testdata['LN_IC50']


        return traindata, valdata, testdata
