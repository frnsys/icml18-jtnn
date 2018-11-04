import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import sys
from tqdm import tqdm
from optparse import OptionParser

from jtnn import Vocab, JTNNVAE, MoleculeDataset
import rdkit

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = OptionParser()
parser.add_option("-t", "--train", dest="train_path")
parser.add_option("-v", "--vocab", dest="vocab_path")
parser.add_option("-s", "--save_dir", dest="save_path")
parser.add_option("-b", "--batch", dest="batch_size", default=40)
parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
parser.add_option("-d", "--depth", dest="depth", default=3)
parser.add_option("-c", "--classes", dest="n_classes", default=0)
parser.add_option("--conditional", action="store_true", dest="conditional")
opts,args = parser.parse_args()

vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)]
vocab = Vocab(vocab)

batch_size = int(opts.batch_size)
hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)
n_classes = int(opts.n_classes)

if opts.conditional and n_classes <= 0:
    print('If the --conditional flag is set, --classes must be > 0. Exiting.')
    sys.exit(1)

model = JTNNVAE(vocab, hidden_size, latent_size, depth, n_classes)

for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant(param, 0)
    else:
        nn.init.xavier_normal(param)

model = model.cuda()
print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
scheduler.step()

dataset = MoleculeDataset(opts.train_path, labeled=opts.conditional)

MAX_EPOCH = 3
PRINT_ITER = 20

for epoch in range(MAX_EPOCH):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=lambda x:x, drop_last=True)

    iter = tqdm(enumerate(dataloader))
    stats = {
        'wacc': [],
        'tacc': [],
        'sacc': [],
        'dacc': []
    }
    for it, batch in iter:
        sizes = []
        for mol_tree in batch:
            if opts.conditional:
                mol_tree, label = mol_tree
            for node in mol_tree.nodes:
                if node.label not in node.cands:
                    node.cands.append(node.label)
                    node.cand_mols.append(node.label_mol)
                sizes.append(len(node.cands))

        print('max size:', max(sizes))

        model.zero_grad()
        loss, kl_div, wacc, tacc, sacc, dacc = model(batch, beta=0, conditional=opts.conditional)
        loss.backward()
        optimizer.step()

        iter.set_postfix(
            ep=epoch,
            kl=kl_div,
            word=wacc, # word accuracy
            topo=tacc, # topo accuracy
            assm=sacc, # assm accuracy
            steo=dacc  # steo accuracy
        )
        stats['wacc'].append(wacc)
        stats['tacc'].append(tacc)
        stats['sacc'].append(sacc)
        stats['dacc'].append(dacc)
        torch.cuda.empty_cache()

    scheduler.step()
    print("learning rate: %.6f" % scheduler.get_lr()[0])
    torch.save(model.state_dict(), opts.save_path + "/model.iter-" + str(epoch))

with open('stats.json', 'w') as f:
    json.dump(stats, f)
