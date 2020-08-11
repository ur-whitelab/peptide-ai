import numpy as np
from sys import argv

if len(argv) != 4 and len(argv) != 5:
    print('Usage: get_auc_average.py [dataset_name] [min_idx] [max_idx] [n_digits (Default: 4)]')
    exit()

dataset_name = argv[1]
min_idx = int(argv[2])
max_idx = int(argv[3])
if len(argv) == 5:
    n_digits = argv[4]
else:
    n_digits = 4

data = np.zeros(max_idx+1 - min_idx)

for i in range(min_idx, max_idx+1):
    points = np.genfromtxt('{}_auc.txt'.format(str(i).zfill(n_digits)))
    if len(points) > 1:
        for point in points:
            data[i] += point
    else:
        data[i] += points

avg_auc = np.mean(data)
stdev_auc = np.std(data)

with open('{}_auc_statistics.txt'.format(dataset_name), 'w+') as f:
    f.write('Average AUC: {}'.format(avg_auc))
    f.write('AUC STDEV: {}'.format(stdev_auc))

