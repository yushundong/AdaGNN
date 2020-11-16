
from __future__ import division
from __future__ import print_function
import argparse
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from adagnn.utils import accuracy, load_data_1_3, load_data_4_6
from adagnn.models import AdaGNN
from adagnn.pre_train_2_layer import pre_train
import sys
print(sys.argv[0])

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--layers', type=int, default=8,
                    help='Layer number of AdaGNN model.')
parser.add_argument('--mode', type=str, default='s',
                    help='Regularization of adjacency matrix in {"r", "s"}.')
parser.add_argument('--dataset', type=str, default='BlogCatalog',
                    help='dataset from {"BlogCatalog", "Flickr", "ACM", "cora", "citeseer", pubmed"}.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,  # 0.0001
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=9e-6,  # 9e-12
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--l1norm', type=float, default=1e-6,  # 1e-6
                    help='L1 loss on Phi in each layer.')
parser.add_argument('--hidden', type=int, default=128,  # 16
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.1,  # 0.2
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.manual_seed(args.seed)

if args.dataset in ['BlogCatalog', 'Flickr', 'ACM']:
    adj, features, labels, idx_train, idx_val, idx_test, gamma, patience = load_data_1_3(args.dataset, args.mode)
elif args.dataset in ['cora', 'citeseer', 'pubmed']:
    adj, features, labels, idx_train, idx_val, idx_test, gamma, patience = load_data_4_6(args.dataset, args.mode)
else:
    print('No such dataset supported !')
    assert 0==1

model = AdaGNN(diag_dimension=features.shape[0], nfeat=features.shape[1],
                                                                 nhid=args.hidden, nlayer=args.layers,
                                                                 nclass=labels.max().item() + 1,
                                                                 dropout=args.dropout)
print(model)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

if args.layers > 2:
    # Pre-train the first & last layer for faster convergence
    pre_train(args.cuda, args.dataset, args.hidden, adj, features, labels, idx_train, idx_val, gamma)
    if args.mode == 's':
        model.load_state_dict(torch.load(args.dataset + '-2.pkl'),
                              strict=False)
    elif args.mode == 'r':
        model.load_state_dict(torch.load(args.dataset + '-2.pkl'),
                              strict=False)

stop_count = 0
val_loss_final = 0
last_loss = 1000

def train(epoch):
    global val_loss_final
    global stop_count
    global last_loss

    t = time.time()
    the_l1 = 0

    for k, v in model.named_parameters():
        if 'learnable_diag' in k:
            the_l1 += torch.sum(abs(v))

    model.train()
    optimizer.zero_grad()
    output = model(features, adj)

    loss_train = F.nll_loss(output[idx_train], labels[idx_train]) + args.l1norm * the_l1
    acc_train = accuracy(output[idx_train], labels[idx_train])

    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val]) + args.l1norm * the_l1
    acc_val = accuracy(output[idx_val], labels[idx_val])

    if loss_val.item() > last_loss:
        stop_count += 1
    else:
        stop_count = 0
    last_loss = loss_val.item()

    if epoch == 0:
        val_loss_final = loss_val.item()
    elif loss_val.item() < val_loss_final:
        val_loss_final = loss_val.item()
        torch.save(model.state_dict(), args.dataset + '-' + str(args.layers) + '.pkl')

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

    if stop_count >= patience:  # 6
        print("Early stop  ! ")

def test():
    try:
        model.load_state_dict(torch.load(args.dataset + '-' + str(args.layers) + '.pkl'),
                              strict=True)
    except FileNotFoundError:
        model.load_state_dict(torch.load(args.dataset + '-' + str(2) + '.pkl'),
                              strict=False)
    model.eval()
    output = model(features, adj)
    if args.dataset != 'citeseer':
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
    if args.dataset == 'citeseer':
        loss_test = F.nll_loss(output[idx_test], labels[-1000:])
        acc_test = accuracy(output[idx_test], labels[-1000:])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

t_total = time.time()
for epoch in range(args.epochs):
    model.train()
    train(epoch)
    if stop_count >= patience:
        break
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

test()
