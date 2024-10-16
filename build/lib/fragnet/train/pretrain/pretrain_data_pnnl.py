import pandas as pd
from dataset import get_pt_dataset, extract_data, save_datasets

tr = pd.read_csv('/people/pana982/solubility/data/full_dataset/isomers/wang/set2/train_new.csv')
vl = pd.read_csv('/people/pana982/solubility/data/full_dataset/isomers/wang/set2/val_new.csv')
ts = pd.read_csv('/people/pana982/solubility/data/full_dataset/isomers/wang/set2/test_new.csv')

df = pd.concat([tr, vl, ts], axis=0)
df.reset_index(drop=True, inplace=True)

ds = get_pt_dataset(df)
ds = extract_data(ds)
save_path = f'pretrain_data/pnnl_iso/ds/train'
save_datasets(ds, save_path)
