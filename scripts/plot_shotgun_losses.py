import numpy as np
import matplotlib.pyplot as plt
from sys import argv

if len(argv) != 4 and len(argv) != 5:
    print('Usage: plot_shotgun_losses.py [dataset_name] [min_idx] [max_idx] [n_digits (Default: 4)]')
    exit()

dataset_name = argv[1]
min_idx = argv[2]
max_idx = argv[3]
if len(argv) == 5:
    n_digits = argv[4]
else:
    n_digits = 4

avg_train_losses = np.zeros(len(np.genfromtxt('{}_train_losses.txt'.format(min_idx.zfill(n_digits)))))
avg_withheld_losses = np.zeros(len(np.genfromtxt('{}_train_losses.txt'.format(min_idx.zfill(n_digits)))))
Z = 0. # normalization constant

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5,9))
plt.title('{}'.format(dataset_name))
plt.xlabel('Training Iteration')
plt.ylabel('Loss')
axes[0].set_ylim(0.0, 1.0)
axes[1].set_ylim(0.0, 1.0)
x_range = range(len(avg_train_losses))
alpha_val = 0.02
small_lw = 0.5
big_lw = 2.0
TRAIN_ITERS_PER_SAMPLE = 50

print('Starting with dataset: {}...'.format(dataset_name))
for i in range(int(min_idx), int(max_idx)):
    Z += 1.
    train_data = np.genfromtxt('{}_train_losses.txt'.format(str(i).zfill(n_digits)))
    axes[0].plot(x_range, train_data, color='blue', alpha=alpha_val, lw=small_lw)
    avg_train_losses += train_data
    # now withheld losses
    withheld_data = np.genfromtxt('{}_withheld_losses.txt'.format(str(i).zfill(n_digits)))
    axes[1].plot(x_range, withheld_data, color='green', alpha=alpha_val, lw=small_lw)
    avg_withheld_losses += withheld_data
avg_train_losses /= Z
avg_withheld_losses /= Z
print('Mean final training loss: {}'.format(avg_train_losses[-1]))
print('Mean final withheld loss: {}'.format(avg_withheld_losses[-1]))
axes[0].plot(x_range, avg_train_losses, color='black', label='Average Training Set Loss', lw=big_lw)
labels = (axes[0].get_xticks())
adjusted_labels = [int(item)//TRAIN_ITERS_PER_SAMPLE for item in labels]
axes[0].set_xticklabels(adjusted_labels)
axes[1].set_xticklabels(adjusted_labels)
axes[0].legend()
axes[1].plot(x_range, avg_withheld_losses, color='black', label='Average Withheld Set Loss', lw=big_lw)
axes[1].legend()
plt.tight_layout()
plt.savefig('{}_average_withheld_loss.png'.format(dataset_name))

np.savetxt('{}_avg_training_loss.txt'.format(dataset_name), avg_train_losses)
np.savetxt('{}_avg_withheld_loss.txt'.format(dataset_name), avg_withheld_losses)
