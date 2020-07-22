import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sys import argv

seaborn.set_context('paper')

if len(argv) != 4 and len(argv) != 5:
    print('Usage: plot_shotgun_losses.py [dataset_name] [min_idx] [max_idx] [control_loss_dir] [n_digits (Default: 4)]')
    exit()

'''This file expects to be executed within a directory where all the loss files from a batch of training runs are. It takes in the min and max indices to read, the name of the dataset (for the plot title), and the name of the directory where the control average training and withheld loss files are to be found. If "None" or "none" is passed for control_loss_dir, instead it will generate those files and save them.'''

dataset_name = argv[1]
min_idx = argv[2]
max_idx = argv[3]
control_dir = argv[4]

if control_dir == 'None' or control_dir == 'none':
    control_dir = None

if len(argv) == 6:
    n_digits = argv[5]
else:
    n_digits = 4

avg_train_losses = np.zeros(np.genfromtxt('{}_train_losses.txt'.format(min_idx.zfill(n_digits))).size)
avg_withheld_accuracy = np.zeros(np.genfromtxt('{}_withheld_accuracy.txt'.format(min_idx.zfill(n_digits))).size)
Z = 0. # normalization constant

fig = plt.figure(figsize=(4,3), dpi=180)
ax = plt.gca()
plt.title('{}'.format(dataset_name))
plt.xlabel('Given Examples')
plt.ylabel('Accuracy')
ax.set_ylim(0.5, 1.0)
x_range = range(len(avg_withheld_accuracy))
alpha_val = 0.1
small_lw = 0.5
big_lw = 1.0

print('Starting with dataset: {}...'.format(dataset_name))
print(x_range, len(avg_withheld_accuracy))
for i in range(int(min_idx), int(max_idx)):
    Z += 1.
    train_data = np.genfromtxt('{}_train_losses.txt'.format(str(i).zfill(n_digits)))
    avg_train_losses += train_data.flatten()
    # now withheld losses
    withheld_data = np.genfromtxt('{}_withheld_accuracy.txt'.format(str(i).zfill(n_digits)))
    ax.plot(x_range, withheld_data, color='C0', alpha=2.*alpha_val, lw=small_lw)
    avg_withheld_accuracy += withheld_data
avg_train_losses /= Z
avg_withheld_accuracy /= Z
print('Mean final training loss: {}'.format(avg_train_losses[-1]))
print('Mean final withheld accuracy: {}'.format(avg_withheld_accuracy[-1]))
labels = (ax.get_xticks())
adjusted_labels = [int(item) for item in labels]
ax.set_xticklabels(adjusted_labels)
ax.plot(x_range, avg_withheld_accuracy, color='C0', label='Average Withheld Set Loss', lw=big_lw, alpha=1 - alpha_val)
#ax.legend()
plt.tight_layout()
plt.savefig('{}.png'.format(dataset_name))
plt.savefig('{}.svg'.format(dataset_name))

np.savetxt('avg_training_loss.txt'.format(dataset_name), avg_train_losses)
np.savetxt('avg_withheld_accuracy.txt'.format(dataset_name), avg_withheld_accuracy)
if control_dir is not None:
    control_avg_train_losses = np.genfromtxt('{}/avg_training_loss.txt'.format(control_dir))
    control_avg_withheld_accuracy = np.genfromtxt('{}/avg_withheld_loss.txt'.format(control_dir))
    ax.plot(x_range, control_avg_train_losses, color='gray', label='Average Control Training Loss', lw=big_lw, ls='--')
    axes[1].plot(x_range, control_avg_withheld_accuracy, color='gray', label='Average Control Withheld Accuracy', lw=big_lw, ls='--')
    ax.legend()
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('{}_vs_RC.png'.format(dataset_name))
