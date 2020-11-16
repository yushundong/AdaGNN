import numpy as np
import scipy.sparse as sp
import torch
from numpy import inf
import scipy.io as scio
import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
import pickle as pkl
import networkx as nx
from numpy import inf


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)



def load_data_1_3(dataset="BlogCatalog", mode='s'):
    print('Loading {} dataset...'.format(dataset))
    dataFile = 'data/' + dataset + '/' + dataset + '.mat'
    data = scio.loadmat(dataFile)
    labels = encode_onehot(list(data['Label'][:, 0]))
    adj = sp.csr_matrix(data['Network'].toarray()[:, :])
    features = data['Attributes'].toarray()[:, :]
    print('Dataset has {} nodes, {} features.'.format(adj.shape[0], features.shape[1]))

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    D = []
    for i in range(adj.sum(axis=1).shape[0]):
        D.append(adj.sum(axis=1)[i, 0])
    D = np.diag(D)
    l = D - adj

    if mode == 's':
        with np.errstate(divide='ignore'):
            D_norm = D ** (-0.5)
        D_norm[D_norm == inf] = 0
        adj = sp.coo_matrix(D_norm.dot(l).dot(D_norm))
    elif mode == 'r':
        with np.errstate(divide='ignore'):
            D_norm = np.linalg.inv(D)
        adj = sp.coo_matrix(D_norm.dot(l))

    list_split = []
    length_of_data = adj.shape[0]
    train_percent = 0.1
    val_percent = 0.2

    for i in range(length_of_data):
        list_split.append(i)

    node_perm = np.random.permutation(labels.shape[0])
    idx_train = node_perm[:int(train_percent * length_of_data)]  # list_split
    idx_val = node_perm[int(train_percent * length_of_data): int(train_percent * length_of_data + val_percent * length_of_data)]
    idx_test = node_perm[int(train_percent * length_of_data + val_percent * length_of_data): ]
    gamma = 0.9
    patience = 6

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, gamma, patience


def load_data_4_6(dataset_str, mode):  # {'pubmed', 'citeseer', 'cora'}

    print("Loading dataset:  " + dataset_str)
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = normalize_features(features)
    features = torch.FloatTensor(np.array(features.todense()))

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + sp.eye(adj.shape[0])
    D = []
    for i in range(adj.sum(axis=1).shape[0]):
        D.append(adj.sum(axis=1)[i, 0])
    D = np.diag(D)
    l = D - adj
    if mode == 'r':
        with np.errstate(divide='ignore'):
            D_norm = np.linalg.inv(D)
        adj = sp.coo_matrix(D_norm.dot(l))
    elif mode == 's':
        with np.errstate(divide='ignore'):
            D_norm = D ** (-0.5)
        D_norm[D_norm == inf] = 0
        adj = sp.coo_matrix(D_norm.dot(l).dot(D_norm))

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    labels = torch.LongTensor(np.where(labels)[1])
    gamma_set = {'cora': 0.1, 'citeseer': 0.9, 'pubmed': 0.7}
    gamma = gamma_set[dataset_str]
    patience = 20
    idx_test = torch.LongTensor(test_idx_range.tolist())
    idx_train = torch.LongTensor(range(len(y)))
    idx_val = torch.LongTensor(range(len(y), len(y) + 500))

    return adj, features, labels, idx_train, idx_val, idx_test, gamma, patience

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    with np.errstate(divide='ignore'):
        r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
    
