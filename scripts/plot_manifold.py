from utils import *
import numpy as np
import matplotlib.pyplot as plt

data_names = [
    'antibacterial',
    'anticancer',
    'antifungal',
    'antiHIV',
    'antiMRSA',
    'antiparasital',
    'antiviral',
    'hemolytic',
    'soluble',
    'shp2',
    'tula2',
    'human',
    'antibacterial-fake',
    'anticancer-fake',
    'antifungal-fake',
    'antiHIV-fake',
    'antiMRSA-fake',
    'antiparasital-fake',
    'antiviral-fake',
    'hemolytic-fake',
    'insoluble',
    'shp2-fake',
    'tula2-fake',
    'human-fake']

seqs = []
labels = []
for i,n in enumerate(data_names):
    s, _ = load_data(os.path.join('active_learning_data', '{}-sequence-vectors.npy'.format(n)), withheld_percent=0.0)
    #s = s[:100]
    seqs.append(s)
    labels.extend([i] * len(s))

seqs = np.concatenate(seqs, axis=0)
lengths = np.sum(seqs, axis=(1,2))
print(lengths.shape)
seqs = seqs / lengths[:, np.newaxis, np.newaxis]
project_peptides('all', seqs, labels, plt.get_cmap('Spectral'), labels=data_names)