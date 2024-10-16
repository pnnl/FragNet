import pickle
import pandas as pd
import torch
from models import FragNetPreTrain
from data import collate_fn_pt as collate_fn
from torch.utils.data import DataLoader
import os
import numpy as np


files = os.listdir('pretrain_data/unimol_exp_/ds/')
data=[]
lens=0
for f in files:

    df1 = pd.read_pickle(f'pretrain_data/unimol_exp_/ds/{f}')
    df2 = pd.read_pickle(f'pretrain_data/unimol_exp/ds/{f}')

    
    for i in range(len(df1)):

        data.append(df1[i].y.item() == df2[i].y.item())

    # break
    lens+=len(df2)

print("non_zero: ", np.count_nonzero(np.array(data)) == len(data) == lens )