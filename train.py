from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import time
import uuid

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import GCNII, DSGCN
from utils import accuracy, load_citation

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=32, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--wd1', type=float, default=0.01, help='Weight decay for graph convolution parameters.')
parser.add_argument('--wd2', type=float, default=5e-4, help='Weight decay for linear layers.')
parser.add_argument('--layer', type=int, default=8, help='Number of graph propagation layers.')
parser.add_argument('--hidden', type=int, default=64, help='Hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate.')
parser.add_argument('--patience', type=int, default=100, help='Patience for early stopping.')
parser.add_argument('--data', default='cora', help='Dataset: citeseer, cora, or pubmed.')
parser.add_argument('--dev', type=int, default=0, help='CUDA device id.')
parser.add_argument('--cpu', action='store_true', default=False, help='Force training on CPU.')
parser.add_argument('--model', type=str, default='dsgcn', choices=['dsgcn', 'gcnii'], help='Backbone model.')
parser.add_argument('--alpha', type=float, default=0.1, help='Initial-feature skip coefficient alpha.')
parser.add_argument('--lamda', type=float, default=0.5, help='Lambda used by GCNII only.')
parser.add_argument('--variant', action='store_true', default=False, help='Use the GCNII variant model.')
parser.add_argument('--eta0', type=float, default=0.8, help='Initial neighborhood aggregation strength eta_0 for DSGCN.')
parser.add_argument('--eta_min', type=float, default=0.1, help='Lower bound eta_min for DSGCN schedule.')
parser.add_argument('--eta_lambda', type=float, default=4.0, help='Decay-rate parameter lambda for DSGCN schedule.')
parser.add_argument('--eta_power', type=float, default=1.5, help='Power parameter p for DSGCN schedule.')
parser.add_argument('--test', action='store_true', default=True, help='Evaluate on the test set.')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.data)
use_cuda = torch.cuda.is_available() and not args.cpu
device = torch.device(f'cuda:{args.dev}' if use_cuda else 'cpu')

features = features.to(device)
adj = adj.to(device)
labels = labels.to(device)
idx_train = idx_train.to(device)
idx_val = idx_val.to(device)
idx_test = idx_test.to(device)

os.makedirs('pretrained', exist_ok=True)
checkpt_file = os.path.join('pretrained', f'{uuid.uuid4().hex}.pt')
print('device:', device)
print('checkpoint:', checkpt_file)

if args.model == 'gcnii':
    model = GCNII(
        nfeat=features.shape[1],
        nlayers=args.layer,
        nhidden=args.hidden,
        nclass=int(labels.max()) + 1,
        dropout=args.dropout,
        lamda=args.lamda,
        alpha=args.alpha,
        variant=args.variant,
    ).to(device)
else:
    model = DSGCN(
        nfeat=features.shape[1],
        nlayers=args.layer,
        nhidden=args.hidden,
        nclass=int(labels.max()) + 1,
        dropout=args.dropout,
        alpha=args.alpha,
        eta0=args.eta0,
        eta_min=args.eta_min,
        eta_lambda=args.eta_lambda,
        eta_power=args.eta_power,
    ).to(device)
    print('DSGCN eta schedule:', [round(v, 4) for v in model.get_eta_schedule()])

optimizer = optim.Adam(
    [
        {'params': model.params1, 'weight_decay': args.wd1},
        {'params': model.params2, 'weight_decay': args.wd2},
    ],
    lr=args.lr,
)


def train():
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    return loss_train.item(), acc_train.item()


@torch.no_grad()
def validate():
    model.eval()
    output = model(features, adj)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    return loss_val.item(), acc_val.item()


@torch.no_grad()
def test():
    model.load_state_dict(torch.load(checkpt_file, map_location=device))
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    return loss_test.item(), acc_test.item()


t_total = time.time()
bad_counter = 0
best = float('inf')
best_epoch = 0
acc = 0.0
for epoch in range(args.epochs):
    loss_tra, acc_tra = train()
    loss_val, acc_val = validate()
    print(
        'Epoch:{:04d}'.format(epoch + 1),
        'train',
        'loss:{:.3f}'.format(loss_tra),
        'acc:{:.2f}'.format(acc_tra * 100),
        '| val',
        'loss:{:.3f}'.format(loss_val),
        'acc:{:.2f}'.format(acc_val * 100),
    )
    if loss_val < best:
        best = loss_val
        best_epoch = epoch
        acc = acc_val
        torch.save(model.state_dict(), checkpt_file)
        bad_counter = 0
    else:
        bad_counter += 1
    if bad_counter == args.patience:
        break

if args.test:
    acc = test()[1]

print('Train cost: {:.4f}s'.format(time.time() - t_total))
print('Load {}th epoch'.format(best_epoch))
print('Test' if args.test else 'Val', 'acc.:{:.1f}'.format(acc * 100))
