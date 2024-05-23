import json

import numpy as np

nkn2024_atomic_num_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 
18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 
42, 44, 46, 47, 48, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 71, 
72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 90, 96, 0]  # 0 is for virtual node.
max_atoms = 38
n_bonds = 4

def one_hot_nkn2024(data, out_size=38):
    num_max_id = len(nkn2024_atomic_num_list)
    assert data.shape[0] == out_size
    b = np.zeros((out_size, num_max_id), dtype=np.float32)
    for i in range(out_size):
        ind = nkn2024_atomic_num_list.index(data[i])
        b[i, ind] = 1.
    return b


def transform_fn_nkn2024(data):
    node, adj, label = data
    # convert to one-hot vector
    # node = one_hot(node).astype(np.float32)
    node = one_hot_nkn2024(node).astype(np.float32)
    # single, double, triple and no-bond. Note that last channel axis is not connected instead of aromatic bond.
    adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)],
                         axis=0).astype(np.float32)
    return node, adj, label


def get_val_ids():
    file_path = '../data/valid_idx_zinc.json'
    print('loading train/valid split information from: {}'.format(file_path))
    with open(file_path) as json_data:
        data = json.load(json_data)
    val_ids = [idx-1 for idx in data]
    return val_ids