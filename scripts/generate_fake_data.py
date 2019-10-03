import numpy as np
from sys import argv

'''Expects a raw sequences file with peptide sequences ONLY.'''

if len(argv) != 2:
    print('Usage: generate_fake_data.py [sequences_file]')
    exit()

filename = argv[1]

with open(filename, 'r') as f:
    lines = f.read().splitlines()

ALPHABET = ['A','R','N','D','C','Q','E','G','H','I', 'L','K','M','F','P','S','T','W','Y','V']
        
def pep_to_int_list(pep):
    '''Takes a single string of amino acids and translates to a list of ints'''
    return(list(map(ALPHABET.index, pep.replace('\n', ''))))

# convert to lists of integers
int_peps = []
for line in lines:
    int_peps.append(pep_to_int_list(line))

# count up the amino acid and length distributions
background_counts = np.zeros(len(ALPHABET))
lengths = [len(item) for item in int_peps]

for pep in int_peps:
    for letter in pep:
        background_counts[letter] += 1
background_dist = background_counts/np.sum(background_counts)

print(np.sum(background_dist))
print(background_counts, '\n', background_dist)

fake_peps = []
# generate a number of dummy peptides equal to the size of the real dataset
for _ in int_peps:
    # get random length of one of the peptides from our dataset
    length = lengths[np.random.randint(0, high=len(lengths))]
    fake_pep = ''
    # fill our fake peptide with letters drawn from the distro of the dataset
    for i in range(length):
        fake_pep += ALPHABET[np.random.choice(len(ALPHABET), p=background_dist)]
    fake_peps.append(fake_pep)
    # print(fake_pep)

with open(filename.split('_')[0] + '_FAKE_' + filename.split('_')[1], 'w+') as f:
    for pep in fake_peps:
        f.write(pep+'\n')

with open(filename.split('_')[0] + '_FAKE_raw_' + filename.split('_')[1], 'w+') as f:
    for i, pep in enumerate(fake_peps):
        f.write(f'FAKE{i:05d}\n')
        f.write(pep+'\n')
