import os
import sys
import torch

from optparse import OptionParser
import rdkit

from jtnn import Vocab, JTNNVAE

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = OptionParser()
parser.add_option("-n", "--nsample", dest="nsample")
parser.add_option("-v", "--vocab", dest="vocab_path")
parser.add_option("-m", "--model", dest="model_path")
parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
parser.add_option("-d", "--depth", dest="depth", default=3)
parser.add_option("-c", "--classes", dest="n_classes", default=0)
parser.add_option("-k", "--class", dest="class_", default=0)
parser.add_option("--conditional", action="store_true", dest="conditional")
opts,args = parser.parse_args()

vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)]
vocab = Vocab(vocab)

hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)
nsample = int(opts.nsample)
n_classes = int(opts.n_classes)

if opts.conditional and n_classes <= 0:
    print('If the --conditional flag is set, --classes must be > 0. Exiting.')
    sys.exit(1)

model = JTNNVAE(vocab, hidden_size, latent_size, depth, n_classes)

print('Loading saved model')
saves = sorted(os.listdir(opts.save_path))
path = os.path.join(opts.save_path, saves[-1])
model.load_state_dict(torch.load(path))
model = model.cuda()

torch.manual_seed(0)
for i in range(nsample):
    print(model.sample_prior(prob_decode=True, class_=opts.class_))
