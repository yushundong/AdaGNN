
from __future__ import division
from __future__ import print_function
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from adagnn.utils import accuracy
from adagnn.models import AdaGNN


def pre_train(cuda, dataset, hidden, adj, features, labels, idx_train, idx_val, gamma):

    stop_count = 0
    val_loss_final = 0
    last_loss = 1000

    model = AdaGNN(diag_dimension=features.shape[0],
                                 nfeat=features.shape[1],
                                 nhid=hidden, nlayer=2,
                                 nclass=labels.max().item() + 1,
                                 dropout=0.5)

    optimizer = optim.Adam(model.parameters(),
                           lr=0.01, weight_decay=9e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=gamma)
    if cuda:
        model.cuda()

    for epoch in range(300):
        t = time.time()
        the_l1 = 0

        for k, v in model.named_parameters():
            if 'learnable_diag' in k:
                the_l1 += torch.sum(abs(v))

        model.train()
        optimizer.zero_grad()
        output = model(features, adj)

        loss_train = F.nll_loss(output[idx_train], labels[idx_train]) + 1e-6 * the_l1
        acc_train = accuracy(output[idx_train], labels[idx_train])

        loss_train.backward()
        optimizer.step()


        loss_val = F.nll_loss(output[idx_val], labels[idx_val]) + 1e-6 * the_l1
        acc_val = accuracy(output[idx_val], labels[idx_val])

        if loss_val.item() > last_loss:
            stop_count += 1
        else:
            stop_count = 0
        last_loss = loss_val.item()

        if epoch == 0:
            val_loss_final = loss_val.item()
        elif loss_val.item() < val_loss_final:  # and epoch >= 100:
            val_loss_final = loss_val.item()
            torch.save(model.state_dict(), dataset + '-' + str(2) + '.pkl')

        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

        if stop_count >= 6 and epoch > 20:
            print("Early stop - pretraining process finished ! ")
            return 0
        scheduler.step()
    print("Pretraining process finished ! ")


