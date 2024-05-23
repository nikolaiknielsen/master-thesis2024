import json

import numpy as np

hmdb_atomic_num_list = [5, 6, 7, 8, 9, 14, 15, 16, 17, 27, 33, 34, 35, 53, 0] # 0 is for virtual node.
max_atoms = 16
n_bonds = 4

def one_hot_hmdb(data, out_size=38):
    num_max_id = len(hmdb_atomic_num_list)
    assert data.shape[0] == out_size
    b = np.zeros((out_size, num_max_id), dtype=np.float32)
    for i in range(out_size):
        ind = hmdb_atomic_num_list.index(data[i])
        b[i, ind] = 1.
    return b

def transform_fn_hmdb(data):
    smiles, spectra, node, adj, label = data
    if isinstance(spectra, str):
        spectra = np.array([float(x) for x in spectra.split(';')], dtype=np.float32)
    if np.isnan(spectra).any():
        return None  #
    # convert to one-hot vector
    # node = one_hot(node).astype(np.float32)
    node = one_hot_hmdb(node).astype(np.float32)
    # single, double, triple and no-bond. Note that last channel axis is not connected instead of aromatic bond.
    adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)],
                         axis=0).astype(np.float32)
    return smiles, spectra, node, adj, label


def get_val_ids():
    file_path = '../data/valid_idx_zinc.json'
    print('loading train/valid split information from: {}'.format(file_path))
    with open(file_path) as json_data:
        data = json.load(json_data)
    val_ids = [idx-1 for idx in data]
    return val_ids