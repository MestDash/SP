import pandas as pd
import numpy as np
from sys import argv
from sklearn.metrics import precision_recall_curve

# We need the aa list, some sequences may have uncommon amino acids
# We also need this for the matrix row index
amino = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

# Load the PSWM and open the training subset as a dataframe
PSWM = np.load('PSWM.npy')
file = str(argv[1])
training = pd.read_csv(file, sep='\t')

# Class extractor, 1 is positive fo SP
# In our dataframe, positive sequence have a float object as SP length,
# while negative sequence have a NaN value
# We can use this to discriminate the class
seq_class = []
for i, row in training.iterrows():
    if pd.isnull(row["End"]) is True:
        seq_class.append(0)
    else:
        seq_class.append(1)


# Def to compute the score of a subsequence of length 15
def score_compute(sub_seq):
    score = 0
    for j in range(len(sub_seq)):
        # index
        aa = sub_seq[j]
        # this accounts for other amino acids
        if aa not in amino:
            score += 0
        else:
            l = int(amino.index(aa))
            score += PSWM[l, j]
    return score


# Def to divide a sequence into subsequences of length 15
def sliding_window(element, result):
    best = -1000
    for k in range(len(element) - 15 + 1):
        sub_seq = element[k:k+15]
        temp = score_compute(sub_seq)
        if temp > best:
            best = temp
    result.append(best)


lst = []
result = []

# Extract the sequences from the input file
for i, row in training.iterrows():
    sequence = str(row["Sequence"])
    lst.append(sequence)

# Compute the scores for each sequence and store the highest scores in a list
for element in lst:
    sliding_window(element, result)

# We can put a condition for giving the optimal threshold when needed

# thresholds: the thresholds values
precision, recall, thresholds = precision_recall_curve(seq_class, result)
# compute f-scores at varying thresholds
fscore = (2 * precision * recall) / (precision + recall)
# get the index of the maximum value of the f-score
index = np.argmax(fscore)
# retrieve the threshold value corresponding to the max f-score computed above
optimal_threshold = thresholds[index]
print(optimal_threshold)
number_of_seqs = len(lst)

# Save the output data in a file
with open('recall_result.txt', 'w') as output:
    output.write('Number of input sequences: ' + str(number_of_seqs) + '\n')
    output.write('Optimal threshold: ' + str(optimal_threshold) + '\n')

