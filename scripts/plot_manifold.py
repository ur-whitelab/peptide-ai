from utils import *
import numpy as np
import matplotlib.pyplot as plt
import tqdm

datasets = load_datasets('active_learning_data/', withheld_percent=0.01)
fig, axs= plt.subplots(nrows=3, ncols=4, figsize=(14,8), sharex=True, sharey=False)

with tqdm.tqdm(total = 3 * 4) as pbar:
    for i in range(3):
        for j in range(4):
            k = i * 4 + j
            pbar.update(1)
            name, (labels, peps), _ = datasets[k]
            project_peptides(name, peps, [np.argmax(x) for x in labels],
                plt.get_cmap('coolwarm'),
                labels=['active', 'inactive'],
                ax=axs[i,j],
                colorbar = False)
plt.tight_layout()
plt.savefig('mannifold_subs.png', dpi=300)