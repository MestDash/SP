import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

amino = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
background = {'A': '0.0825', 'R': '0.0553', 'N': '0.0406', 'D': '0.0546',
              'C': '0.0138', 'Q': '0.0393', 'E': '0.0672', 'G': '0.0707',
              'H': '0.0227', 'I': '0.0591', 'L': '0.0965', 'K': '0.058',
              'M': '0.0241', 'F': '0.0386', 'P': '0.0474', 'S': '0.0665',
              'T': '0.0536', 'W': '0.011', 'Y': '0.0292', 'V': '0.0685'}


def pswm_train1(training_sp, matrices):
    sites = []
    for i, row in training_sp.iterrows():
        sp_length = int(row["End"])
        if sp_length >= 13:
            sequence = str(row["Sequence"])
            site = sequence[sp_length - 13:sp_length + 2]
            sites.append(site)
    columns = 15
    rows = 20
    PSPM = np.ones((rows, columns), dtype=float)
    for sequence in sites:
        for j in range(15):
            for i in range(20):
                if sequence[j] == amino[i]:
                    PSPM[i, j] += 1
    i = 0
    for key in background:
        PSPM[i] = PSPM[i] / (float(background[key]) * (20 + len(sites)))
        i += 1
    PSWM = np.log2(PSPM)
    PSWM_list = PSWM.tolist()
    for i in range(20):
        for j in range(15):
            PSWM_list[i][j] = round(PSWM_list[i][j], 3)
    matrices.append(PSWM_list)
    return PSWM


def pswm_train(training_sp, matrices):
    sites = []
    for i, row in training_sp.iterrows():
        sp_length = int(row["End"])
        if sp_length >= 13:
            sequence = str(row["Sequence"])
            tot = sequence.count('X') + sequence.count('U') + sequence.count('O') + sequence.count('J') + sequence.count('Z') + sequence.count('B')
            if tot == 0:
                site = sequence[sp_length - 13:sp_length + 2]
                sites.append(site)
    columns = 15
    rows = 20
    PSPM = np.ones((rows, columns), dtype=float)
    for sequence in sites:
        for j in range(15):
            for i in range(20):
                if sequence[j] == amino[i]:
                    PSPM[i, j] += 1
    i = 0
    for key in background:
        PSPM[i] = PSPM[i] / (float(background[key]) * (20 + len(sites)))
        i += 1
    PSWM = np.log2(PSPM)
    PSWM_list = PSWM.tolist()
    for i in range(20):
        for j in range(15):
            PSWM_list[i][j] = round(PSWM_list[i][j], 3)
    matrices.append(PSWM_list)
    return PSWM


def score_compute1(sub_seq, pswm):
    score = 0
    for j in range(len(sub_seq)):
        # index
        aa = sub_seq[j]
        # this accounts for other amino acids
        if aa not in amino:
            score += 0
        else:
            l = int(amino.index(aa))
            score += pswm[l, j]
    return score


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


# we constrain the search to the first 90 residues
def sliding_window(element, result, pswm):
    best = -1000
    for k in range(76):
        sub_seq = element[k:k+15]
        temp = score_compute(sub_seq, pswm)
        if temp > best:
            best = temp
    result.append(best)


def threshold1(opt, pswm, k, report):
    seq_class = []
    lst = []
    for i, row in opt.iterrows():
        sequence = str(row["Sequence"])
        lst.append(sequence)
        if pd.isnull(row["End"]) is True:
            seq_class.append(0)
        else:
            seq_class.append(1)
    result = []
    for element in lst:
        sliding_window(element, result, pswm)
    precision, recall, thresholds = precision_recall_curve(seq_class, result)
    fscore = (2 * precision * recall) / (precision + recall)
    index = np.argmax(fscore)
    optimal_threshold = thresholds[index]
    number_of_seqs = len(lst)
    string = f'Optimization set: subset{k+1}' + '\n' +\
             'Number of input sequences: ' +\
             str(number_of_seqs) + '\n' +\
             'Optimal threshold: ' +\
             str(optimal_threshold) + '\n'
    report.append(string)
    return optimal_threshold


def threshold(opt, pswm, k, report):
    seq_class = []
    lst = []
    for i, row in opt.iterrows():
        sequence = str(row["Sequence"])
        tot = sequence.count('X') + sequence.count('U') + sequence.count('O') + sequence.count('J') + sequence.count('Z') + sequence.count('B')
        if tot == 0:
            lst.append(sequence)
            if pd.isnull(row["End"]) is True:
                seq_class.append(0)
            else:
                seq_class.append(1)
    result = []
    for element in lst:
        sliding_window(element, result, pswm)
    precision, recall, thresholds = precision_recall_curve(seq_class, result)
    fscore = (2 * precision * recall) / (precision + recall)
    index = np.argmax(fscore)
    optimal_threshold = thresholds[index]
    number_of_seqs = len(lst)
    string = f'Optimization set: subset{k+1}' + '\n' +\
             'Number of input sequences: ' +\
             str(number_of_seqs) + '\n' +\
             'Optimal threshold: ' +\
             str(optimal_threshold) + '\n'
    report.append(string)
    return optimal_threshold


def prediction1(valid, pswm, optimal_threshold, report, k):
    lst = []
    seq_class = []
    for i, row in valid.iterrows():
        sequence = str(row["Sequence"])
        lst.append(sequence)
        if pd.isnull(row["End"]) is True:
            seq_class.append(0)
        else:
            seq_class.append(1)
    result = []
    for element in lst:
        sliding_window(element, result, pswm)
    pred_test = [int(t_s >= optimal_threshold) for t_s in result]
    number_of_seqs = len(lst)
    tn, fp, fn, tp = confusion_matrix(seq_class, pred_test).ravel()
    string = f'Validation set: subset{((k+1)%5) + 1}' + '\n' +\
             'Number of input sequences: ' + str(number_of_seqs) + '\n' +\
             'Confusion matrix:  \n' + 'TN\t\tFP\t\tFN\t\tTP\n' +\
             str(tn) + '\t' + str(fp) + '\t\t' + str(fn) + '\t\t' + str(tp) + '\n' + \
             'Precision: ' + str(precision_score(seq_class, pred_test)) + '\n' + \
             'Accuracy: ' + str(accuracy_score(seq_class, pred_test)) + '\n' + \
             'MCC: ' + str(matthews_corrcoef(seq_class, pred_test)) + '\n'
    report.append(string)


def prediction(valid, pswm, optimal_threshold, report, k):
    lst = []
    seq_class = []
    for i, row in valid.iterrows():
        sequence = str(row["Sequence"])
        tot = sequence.count('X') + sequence.count('U') + sequence.count('O') + sequence.count('J') + sequence.count('Z') + sequence.count('B')
        if tot == 0:
            lst.append(sequence)
            if pd.isnull(row["End"]) is True:
                seq_class.append(0)
            else:
                seq_class.append(1)
    result = []
    for element in lst:
        sliding_window(element, result, pswm)
    pred_test = [int(t_s >= optimal_threshold) for t_s in result]
    number_of_seqs = len(lst)
    tn, fp, fn, tp = confusion_matrix(seq_class, pred_test).ravel()
    string = f'Validation set: subset{((k+1)%5) + 1}' + '\n' +\
             'Number of input sequences: ' + str(number_of_seqs) + '\n' +\
             'Confusion matrix:  \n' + 'TN\t\tFP\t\tFN\t\tTP\n' +\
             str(tn) + '\t' + str(fp) + '\t\t' + str(fn) + '\t\t' + str(tp) + '\n' + \
             'Precision: ' + str(precision_score(seq_class, pred_test)) + '\n' + \
             'Accuracy: ' + str(accuracy_score(seq_class, pred_test)) + '\n' + \
             'MCC: ' + str(matthews_corrcoef(seq_class, pred_test)) + '\n'
    report.append(string)
