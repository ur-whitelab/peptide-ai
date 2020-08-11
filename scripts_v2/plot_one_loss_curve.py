import numpy as np
import matplotlib.pyplot as plt
from sys import argv

if len(argv) != 3 and len(argv) != 4:
    print('Usage: plot_shotgun_losses.py [dataset_name] [run_number] [n_digits (Default: 4)]')
    exit()

dataset_name = argv[1]
run_number = argv[2]
if len(argv) == 4:
    n_digits = argv[3]
else:
    n_digits = 4

TRAIN_ITERS_PER_SAMPLE = 50

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5,9))
plt.title('{}'.format(dataset_name))
plt.xlabel('Number of Observations')
plt.ylabel('Loss')
axes[0].set_ylim(0., 1.0)
axes[1].set_ylim(0.0, 1.0)
train_data = np.genfromtxt('{}_train_losses.txt'.format(run_number.zfill(n_digits)))
x_range = range(len(train_data))
axes[0].plot(x_range, train_data, color='blue', lw=0.5, label='Training Loss')
labels = (axes[0].get_xticks())
adjusted_labels = [int(item)//TRAIN_ITERS_PER_SAMPLE for item in labels]
axes[0].set_xticklabels(adjusted_labels)
withheld_data = np.genfromtxt('{}_withheld_losses.txt'.format(run_number.zfill(n_digits)))
axes[1].plot(x_range, withheld_data, color='green', lw=0.5, label='Withheld Loss')
print('Final training loss: {}'.format(train_data[-1]))
print('Final withheld loss: {}'.format(withheld_data[-1]))
axes[1].set_xticklabels(adjusted_labels)
axes[0].legend()
axes[1].legend()
plt.tight_layout()
plt.savefig('{}_run_{}_withheld_loss.png'.format(dataset_name, run_number))
