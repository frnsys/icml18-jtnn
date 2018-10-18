import rdkit
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles
import sascorer
import sys

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

smiles = []
for line in sys.stdin:
    smiles.append(line.strip())

targets = []
for i in range(len(smiles)):
    logp = Descriptors.MolLogP(MolFromSmiles(smiles[ i ]))
    sa = sascorer.calculateScore(MolFromSmiles(smiles[ i ]))
    targets.append(logp - sa)

smiles = zip(smiles, targets)
smiles = sorted(smiles, key=lambda x:x[1])
for x,y in smiles:
    print(x,y)
