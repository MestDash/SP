import pandas as pd
import numpy as np
from sys import argv
import csv

# Open the file and create a dataframe, put the positive sequence in a separate dataframe
file = str(argv[1])
training = pd.read_csv(file, sep='\t')
training_SP = training[~pd.isna(training['End'])]

# For every positive sequence, we select a portion of the signal peptide (-13, +2)
# The portions are stored in a list, they will function as a 'stacked alignment'
sites = []
for i, row in training_SP.iterrows():
    sp_length = int(row["End"])
    if sp_length >= 13:
        sequence = str(row["Sequence"])
        site = sequence[sp_length - 13:sp_length + 2]
        sites.append(site)

# Set the dimensions of the matrix
columns = len(sites[0])
rows = 20

# Initialize a matrix of size rows by columns filled with ones (pseudocounts) using numpy
PSPM = np.ones((rows, columns), dtype=float)

# For the rows, set the order of amino acids, from top to bottom
amino = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

# Compute the PSPM, 'i' will be the row index and 'j' will be the column index
# This loop scans every site, then adds +1 to the correct amino acid (row, i)
# based on the sequence position (column, j)
for sequence in sites:
    for j in range(15):
        for i in range(20):
            if sequence[j] == amino[i]:
                PSPM[i, j] += 1

# To compute the PSWM, divide every value in the matrix by 20 + N,
# where N is the length of our sequences
# Then, every row of the matrix (corresponds to an aa) needs to be divided
# by the corresponding background composition
background = {'A': '0.0825', 'R': '0.0553', 'N': '0.0406', 'D': '0.0546',
              'C': '0.0138', 'Q': '0.0393', 'E': '0.0672', 'G': '0.0707',
              'H': '0.0227', 'I': '0.0591', 'L': '0.0965', 'K': '0.058',
              'M': '0.0241', 'F': '0.0386', 'P': '0.0474', 'S': '0.0665',
              'T': '0.0536', 'W': '0.011', 'Y': '0.0292', 'V': '0.0685'}

i = 0
for key in background:
    PSPM[i] = PSPM[i] / (float(background[key]) * (20 + len(sites)))
    i += 1

# Log-transform the whole matrix (base e or 2 are fine)
PSWM = np.log2(PSPM)

# Save the PSWM as a npy (binary) file to be used later
np.save('PSWM', PSWM)

# readable form
PSWM_list = PSWM.tolist()
for i in range(20):
    for j in range(15):
        PSWM_list[i][j] = round(PSWM_list[i][j], 3)


with open('matrix.tsv', 'w', newline='') as file:
    # Create a CSV writer with tab as the delimiter
    writer = csv.writer(file, delimiter='\t')
    # Write the data to the TSV file row by row
    for row in PSWM_list:
        writer.writerow(row)