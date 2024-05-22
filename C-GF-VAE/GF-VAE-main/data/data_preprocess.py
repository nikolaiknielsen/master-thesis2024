import os
import sys
# for linux env.
sys.path.insert(0,'..')
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import time
from data.data_frame_parser import DataFrameParser
from data.data_loader import NumpyTupleDataset
from data.smile_to_graph import GGNNPreprocessor


def parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_name', type=str, default='qm9',
                        choices=['qm9', 'zinc250k', 'molu2024', 'nkn2024', 'hmdb'],
                        help='dataset to be downloaded')
    parser.add_argument('--data_type', type=str, default='relgcn',
                        choices=['gcn', 'relgcn'],)
    args = parser.parse_args()
    return args


start_time = time.time()
args = parse()
data_name = args.data_name
data_type = args.data_type
print('args', vars(args))

if data_name == 'qm9':
    max_atoms = 9
elif data_name == 'zinc250k':
    max_atoms = 38
elif data_name == 'molu2024':
    max_atoms = 38
elif data_name == 'nkn2024':
    max_atoms = 38
elif data_name == 'hmdb':
    max_atoms = 38
else:
    raise ValueError("[ERROR] Unexpected value data_name={}".format(data_name))


if data_type == 'relgcn':
    # preprocessor = GGNNPreprocessor(out_size=max_atoms, kekulize=True, return_is_real_node=False)
    preprocessor = GGNNPreprocessor(out_size=max_atoms, kekulize=True)
else:
    raise ValueError("[ERROR] Unexpected value data_type={}".format(data_type))

data_dir = "."
os.makedirs(data_dir, exist_ok=True)

if data_name == 'qm9':
    print('Preprocessing qm9 data:')
    df_qm9 = pd.read_csv('qm9.csv', index_col=0)
    labels = ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2',
              'zpve', 'U0', 'U', 'H', 'G', 'Cv']
    parser = DataFrameParser(preprocessor, labels=labels, smiles_col='SMILES1')
    result = parser.parse(df_qm9, return_smiles=True)
    dataset = result['dataset']
    smiles = result['smiles']
elif data_name == 'zinc250k':
    print('Preprocessing zinc250k data')
    # dataset = datasets.get_zinc250k(preprocessor)
    df_zinc250k = pd.read_csv('zinc250k.csv', index_col=0)
    # Caution: Not reasonable but used in used in chain_chemistry\datasets\zinc.py:
    # 'smiles' column contains '\n', need to remove it.
    # Here we do not remove \n, because it represents atom N with single bond
    labels = ['logP', 'qed', 'SAS']
    parser = DataFrameParser(preprocessor, labels=labels, smiles_col='smiles')
    result = parser.parse(df_zinc250k, return_smiles=True)
    dataset = result['dataset']
    smiles = result['smiles']
    ### FOR THE NEW DATASET 2024 (HAS SAME FORMAT AS zinc250k)
elif data_name == 'molu2024': 
    print('Preprocessing molu2024 data')
    df_molu2024= pd.read_csv('molu2024.csv', index_col=0)
    labels = ['logP', 'qed', 'SAS']
    parser = DataFrameParser(preprocessor, labels=labels, smiles_col='smiles')
    result = parser.parse(df_molu2024, return_smiles=True)
    dataset = result['dataset']
    smiles = result['smiles']
elif data_name == 'nkn2024': 
    print('Preprocessing nkn2024 data')
    df_nkn2024= pd.read_csv('nkn2024.csv', index_col=0)
    labels = ['logP', 'qed', 'SAS']
    parser = DataFrameParser(preprocessor, labels=labels, smiles_col='smiles')
    result = parser.parse(df_nkn2024, return_smiles=True)
    dataset = result['dataset']
    smiles = result['smiles']
elif data_name == 'hmdb':
    print('Preprocessing hmdb data')
    df_hmdb= pd.read_csv('hmdb.csv', index_col=0)
    labels = ['logP', 'qed', 'SAS']
    parser = DataFrameParser(preprocessor, labels=labels, smiles_col='smiles')
    result = parser.parse(df_hmdb, return_smiles=True)
    dataset = result['dataset']
    smiles = result['smiles']
else:
    raise ValueError("[ERROR] Unexpected value data_name={}".format(data_name))

NumpyTupleDataset.save(os.path.join(data_dir, '{}_{}_kekulized_ggnp.npz'.format(data_name, data_type)), dataset)
print('Total time:', time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time)) )

print(smiles[0])
print(smiles.shape)

#### FOR SAVING THE SMILES (ADDED IN) #### 
smiles_df = pd.DataFrame(smiles, columns=['SMILES'])

# Define the CSV file name based on the dataset name
csv_file_name = f'{data_name}_smiles.csv'

# Save the DataFrame to a CSV file
smiles_df.to_csv(csv_file_name, index=False)

print(f'SMILES information saved to {csv_file_name}')
