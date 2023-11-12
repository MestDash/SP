import pandas as pd
import numpy as np
from sys import argv

from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score


amino = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

PSWM = np.load('PSWM.npy')
filename = str(argv[1])
test_set = pd.read_csv(filename, sep='\t')

seq_class = []
ids = []
for i, row in test_set.iterrows():
    sequence = str(row["Sequence"])
    tot = sequence.count('X') + sequence.count('U') + sequence.count('O') + sequence.count('J') + sequence.count('Z') + sequence.count('B')
    if tot == 0:
        ids.append(row["Accession code"])
        if pd.isnull(row["End"]) is True:
            seq_class.append(0)
        else:
            seq_class.append(1)


# Def to compute the score of a subsequence of length 15
def score_compute(sub_seq, pswm):
    score = 0
    tot = sub_seq.count('X') + sub_seq.count('U') + sub_seq.count('O') + sub_seq.count('J') + sub_seq.count('Z') + sub_seq.count('B')
    if tot == 0:
        for j in range(len(sub_seq)):
            # index
            aa = sub_seq[j]
            # this accounts for other amino acids
            l = int(amino.index(aa))
            score += pswm[l, j]
    return score


# Def to divide a sequence into subsequences of length 15
def sliding_window(element, result):
    best = -1000
    for k in range(76):
        sub_seq = element[k:k+15]
        temp = score_compute(sub_seq, PSWM)
        if temp > best:
            best = temp
    result.append(best)


lst = []
result = []

# Extract the sequences
for i, row in test_set.iterrows():
    sequence = str(row["Sequence"])
    tot = sequence.count('X') + sequence.count('U') + sequence.count('O') + sequence.count('J') + sequence.count('Z') + sequence.count('B')
    if tot == 0:
        lst.append(sequence)

# Compute the scores for each sequence and store the highest scores in a list
for element in lst:
    sliding_window(element, result)

optimal_threshold = float(argv[2])
# classify examples in the testing set
pred_test = [int(t_s >= optimal_threshold) for t_s in result]

# Some useful data to print
number_of_seqs = len(lst)
tn, fp, fn, tp = confusion_matrix(seq_class, pred_test).ravel()
print(tn, fp, fn, tp)
print(matthews_corrcoef(seq_class, pred_test))
print(precision_score(seq_class, pred_test))


fp_id = []
fn_id = []
for i in range(len(pred_test)):
    if pred_test[i] == 1 and seq_class[i] == 0:
        fp_id.append(ids[i])
    elif pred_test[i] == 0 and seq_class[i] == 1:
        fn_id.append(ids[i])

with open('FP.txt', 'w') as file:
    for element in fp_id:
        file.write(str(element) + '\n')

with open('FN.txt', 'w') as file:
    for element in fn_id:
        file.write(str(element) + '\n')


# Save the output data in a file
with open('predict_result.txt', 'w') as output:
    output.write('Threshold = ' + str(optimal_threshold) + '\n')
    output.write('Number of input sequences: ' + str(number_of_seqs) + '\n')
    output.write('Confusion matrix:  \n')
    output.write('TN\t\tFP\t\tFN\t\tTP\n')
    output.write(str(tn) + '\t' + str(fp) + '\t\t' + str(fn) + '\t\t' + str(tp) + '\n')
    output.write('Precision: ' + str(precision_score(seq_class, pred_test)) + '\n')
    output.write('Accuracy: ' + str(accuracy_score(seq_class, pred_test)) + '\n')
    output.write('MCC: ' + str(matthews_corrcoef(seq_class, pred_test)))
