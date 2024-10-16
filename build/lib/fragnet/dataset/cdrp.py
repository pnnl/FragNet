import os

from .dataset import FinetuneDataCDRP
from .ext_data_utils.deepttc import DataEncoding
from sklearn.model_selection import train_test_split
from .utils import extract_data, save_datasets, save_ds_parts

def create_cdrp_dataset(args):
        
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    print('args.use_genes1: ', args.use_genes)

    obj = DataEncoding(args.data_dir)

    print('args.use_genes: ', args.use_genes)

    dataset = FinetuneDataCDRP(target_name=args.target_name, data_type=args.data_type, 
                               args=args,
                               use_genes=args.use_genes)


    traindata, testdata = obj.Getdata.ByCancer(random_seed=1, test_size=0.05)
    traindata, valdata = train_test_split(traindata, test_size=.1)

    traindata, valdata, testdata = obj.encode2(
        traindata=traindata, valdata=valdata, testdata=testdata)
    
    traindata.to_csv(f'{args.output_dir}/train.csv', index=False)
    valdata.to_csv(f'{args.output_dir}/val.csv', index=False)
    testdata.to_csv(f'{args.output_dir}/test.csv', index=False)


    if args.save_parts == 0:


        ds = dataset.get_ft_dataset(traindata)
        ds = extract_data(ds)
        print("ds: ", ds[0])
        save_path = f'{args.output_dir}/train'
        save_datasets(ds, save_path)


        ds = dataset.get_ft_dataset(valdata)
        ds = extract_data(ds)
        save_path = f'{args.output_dir}/val'
        save_datasets(ds, save_path)
        
        
        ds = dataset.get_ft_dataset(testdata)
        ds = extract_data(ds)
        save_path = f'{args.output_dir}/test'
        save_datasets(ds, save_path)

    else:
        save_ds_parts(data_creater=dataset, ds=traindata, output_dir=args.output_dir, fold='train')
        save_ds_parts(data_creater=dataset, ds=valdata, output_dir=args.output_dir, fold='val')
        save_ds_parts(data_creater=dataset, ds=testdata, output_dir=args.output_dir, fold='test')

