from torch.utils.data import Dataset
from .mol_tree import MolTree
import numpy as np

class MoleculeDataset(Dataset):

    def __init__(self, data_file, labeled=False):
        self.labeled = labeled
        with open(data_file) as f:
            self.data = []
            for line in f:
                line = line.strip().split('\t')
                smiles = line[0]
                if labeled:
                    smiles, label = line[0], line[1]
                    self.data.append((smiles, label))
                else:
                    smiles = line[0]
                    self.data.append(smiles)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.labeled:
            smiles, label = sample
        else:
            smiles = sample
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        mol_tree.assemble()
        if self.labeled:
            return mol_tree, sample
        else:
            return mol_tree

class PropDataset(Dataset):

    def __init__(self, data_file, prop_file):
        self.prop_data = np.loadtxt(prop_file)
        with open(data_file) as f:
            self.data = [line.strip("\r\n ").split()[0] for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx]
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        mol_tree.assemble()
        return mol_tree, self.prop_data[idx]

