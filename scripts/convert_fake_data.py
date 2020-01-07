import numpy as np
from sys import argv

'''Converts old fake distribution .npy file to new one. Expects two .npy files as input. First
   is the one whose length distribution to match. Second is the one whose amino acid distribution
   to match. Saves over previous fake file with expected naming convention.'''

if len(argv) != 3:
    print('Usage: convert_fake_data.py [length_target_filename] [distribution_target_filename]')
    exit(1)

length_distro_filename = argv[1]
aa_distro_filename = argv[2]

print('Converting fake dataset for {} to use amino acids distributed like {}'.format(
        length_distro_filename, aa_distro_filename))

# these are one-hot encoded already
length_distro_peps = np.load(open(length_distro_filename))
aa_distro_peps = np.load(open(aa_distro_filename))

# since they're already one-hots, sum each 2D array to get peptide lengths
lengths = np.sum(length_distro_peps, axis=(1,2))

# revert from one-hot to alphabet indices (final dimension)
collapsed_aa_peps = np.argwhere(aa_distro_peps)[:,2]

# count unique elements. conveniently np.unique sorts these
_, counts = np.unique(collapsed_aa_peps, return_counts=True)

# cast to float and normalize
background_distro = counts / np.sum(counts, dtype=np.float32)

# make same number and size of fake peptides
fake_peps = np.zeros_like(length_distro_peps)

# generate fake one-hots
for fake in fake_peps:
    # sample a length
    this_length = np.random.choice(lengths)
    for i in range(int(this_length)):
        # sample an index
        this_aa = np.random.choice(len(counts), p=background_distro)
        fake[i][this_aa] += 1

# save over old files
np.save('{}-fake-sequence-vectors.npy'.format(length_distro_filename.split('-')[0]), fake_peps)
