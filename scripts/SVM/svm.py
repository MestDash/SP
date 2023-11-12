from functions import *
import pandas as pd
from sys import argv
from sklearn import svm
from sklearn.metrics import matthews_corrcoef

'''
Training data:
1. SP length (find the cleavage position and calculate the SP length)
2. SP sequence composition
We can combine the data above: each protein is encoded with a 20-dimensional vector corresponding to the composition
of the first K residues in the protein
3. Hydrophobicity, average and maximal hydrophobicity computed over the K N-terminal residues using a 5-residue sliding-window approach
4. Normalized position of the maximal HP
5. Abundance/number of positively charged residues in the K N-terminal residues
for each protein [ sp length, [composition vector], 
'''

sub1 = pd.read_csv(str(argv[1]), sep='\t')
sub2 = pd.read_csv(str(argv[2]), sep='\t')
sub3 = pd.read_csv(str(argv[3]), sep='\t')
sub4 = pd.read_csv(str(argv[4]), sep='\t')
sub5 = pd.read_csv(str(argv[5]), sep='\t')

subsets = [sub1, sub2, sub3, sub4, sub5]
report = []
k = [20, 21, 22, 23, 24]
c = [1, 2, 4, 8]
gamma = [0.5, 1, 2, 'scale']


for i in range(5):
    optim = subsets[i]
    valid = subsets[(i+1)%5]
    training = pd.concat((subsets[(i+2)%5], subsets[(i+3)%5], subsets[(i+4)%5]))

    # Class
    # tr_class = get_class(training)
    # opt_class = get_class(optim)
    val_class = get_class(valid)

    # SVC
    best = 0
    result = []
    for length in k:
        tr_feat, tr_class = get_features_class(training, length)
        opt_feat, opt_class = get_features_class(optim, length)
        # valid_feat = get_features(valid, length)
        for element in gamma:
            for penalty in c:
                mySVC = svm.SVC(C=penalty, kernel='rbf', gamma=element)
                mySVC.fit(tr_feat, tr_class)
                opt_pred_class = mySVC.predict(opt_feat)
                MCC = matthews_corrcoef(opt_class, opt_pred_class)
                print(penalty, element, length, MCC)
                if MCC >= best:
                    best = MCC
                    result = [penalty, element, length]


    # testing
    tr_feat = get_features(training, result[2])
    valid_feat = get_features(valid, result[2])
    mySVC = svm.SVC(C=result[0], kernel='rbf', gamma=result[1])
    mySVC.fit(tr_feat, tr_class)
    val_pred_class = mySVC.predict(valid_feat)
    MCC_test = matthews_corrcoef(val_class, val_pred_class)
    if result[1] != 'scale':
        report.append(str(str((i+2)%5) + str((i+3)%5) + str((i+4)%5) + '\t\t' + str(i) + '\t' + str((i+1)%5) + '\t'+ str(result[0]) + '\t' + str(result[1]) + '\t\t' + str(result[2]) + '\t' + str(round(MCC_test, 3))))
    else:
        report.append(str(str((i+2)%5) + str((i+3)%5) + str((i+4)%5) + '\t\t' + str(i) + '\t' + str((i+1)%5) + '\t'+ str(result[0]) + '\t' + str(result[1]) + '\t' + str(result[2]) + '\t' + str(round(MCC_test, 3))))


with open('result.txt', 'w') as file:
    header = 'train\topt\tval\tC\tgamma\tK\tMCC\n'
    file.write(header)
    for element in report:
        file.write(element + '\n')

'''
The optimal parameters seem to be C=2, gamma=2, K=20
'''
