from sys import argv
import csv
from functions import *

sub1 = pd.read_csv(str(argv[1]), sep='\t')
sub2 = pd.read_csv(str(argv[2]), sep='\t')
sub3 = pd.read_csv(str(argv[3]), sep='\t')
sub4 = pd.read_csv(str(argv[4]), sep='\t')
sub5 = pd.read_csv(str(argv[5]), sep='\t')

subsets = [sub1, sub2, sub3, sub4, sub5]
report = []
matrices = []

for i in range(5):
    optim = subsets[i]
    valid = subsets[(i+1)%5]
    training = pd.concat((subsets[(i+2)%5], subsets[(i+3)%5], subsets[(i+4)%5]))
    training_SP = training[~pd.isna(training['End'])]
    PSWM = pswm_train(training_SP, matrices)
    optimal_threshold = threshold(optim, PSWM, i, report)
    prediction(valid, PSWM, optimal_threshold, report, i)

with open('matrices.tsv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    for element in matrices:
        file.write('\n')
        for row in element:
            writer.writerow(row)

with open('result.txt', 'w') as file:
    for element in report:
        file.write(element + '\n')

