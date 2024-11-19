python3.11 -m venv ~/.env/fragnet
source  ~/.env/fragnet/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
pip install .

mkdir -p fragnet/finetune_data/moleculenet/esol/raw/

wget -O fragnet/finetune_data/moleculenet/esol/raw/delaney-processed.csv https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv

python fragnet/data_create/create_pretrain_datasets.py --save_path fragnet/pretrain_data/esol --data_type exp1s --maxiters 500 --raw_data_path fragnet/finetune_data/moleculenet/esol/raw/delaney-processed.csv

python fragnet/data_create/create_finetune_datasets.py --dataset_name moleculenet --dataset_subset esol --use_molebert True --output_dir fragnet/finetune_data/moleculenet_exp1s --data_dir fragnet/finetune_data/moleculenet --data_type exp1s
