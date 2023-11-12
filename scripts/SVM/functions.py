import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing as pre

amino = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
background = {'A': '0.0825', 'R': '0.0553', 'N': '0.0406', 'D': '0.0546',
              'C': '0.0138', 'Q': '0.0393', 'E': '0.0672', 'G': '0.0707',
              'H': '0.0227', 'I': '0.0591', 'L': '0.0965', 'K': '0.058',
              'M': '0.0241', 'F': '0.0386', 'P': '0.0474', 'S': '0.0665',
              'T': '0.0536', 'W': '0.011', 'Y': '0.0292', 'V': '0.0685'}
hphob = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
         'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
         'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
         'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
         'X': 0}
ahelix = {'A': 1.420, 'R': 0.980, 'N': 0.670, 'D': 1.010, 'C': 0.700,
          'Q': 1.110, 'E': 1.510, 'G': 0.570, 'H': 1.000, 'I': 1.080,
          'L': 1.210, 'K': 1.160, 'M': 1.450, 'F': 1.130, 'P': 0.570,
          'S': 0.770, 'T': 0.830, 'W': 1.080, 'Y': 0.690, 'V': 1.060,
          'X': 0}


def pswm_train(training_sp, matrices):
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


def score_compute(sub_seq, pswm):
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


# we constraint the search to the first 90 residues
def sliding_window(element, result, pswm):
    best = -1000
    for k in range(76):
        sub_seq = element[k:k+15]
        temp = score_compute(sub_seq, pswm)
        if temp > best:
            best = temp
    result.append(best)


def threshold(opt, pswm, k, report):
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


def prediction(valid, pswm, optimal_threshold, report, k):
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


def scale(value):
    val_scaled = round(((value + 4.5) / 9), 3)
    return val_scaled


# def scale1(value):
#     val_scaled = round(((value + 4.5) / 9), 3)
#     return val_scaled


def get_class(df):
    seq_class = []
    for p, row in df.iterrows():
        if pd.isnull(row["End"]) is True:
            seq_class.append(0)
        else:
            seq_class.append(1)
    return np.array(seq_class)


def get_class1(df):
    seq_class = []
    for j, row in df.iterrows():
        sequence = str(row["Sequence"])
        tot = sequence.count('X') + sequence.count('U') + sequence.count('O') + sequence.count('J') + sequence.count('Z') + sequence.count('B')
        if tot == 0:
            if pd.isnull(row["End"]) is True:
                seq_class.append(0)
            else:
                seq_class.append(1)
    return np.array(seq_class)



def get_features(df, length):
    # This vector will contain every feature regarding our sequence
    vector = []
    # we extract all the features we need
    # aa composition and hydrophobicity
    hphob = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
             'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
             'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
             'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
    for j, row in df.iterrows():
        sequence = str(row["Sequence"])
        signal_peptide = sequence[:length]
        for letter in signal_peptide:
            if letter not in amino:
                signal_peptide = signal_peptide.replace(letter, 'X')

        training_composition = {}
        for aa in amino:
            training_composition[aa] = training_composition.get(aa, 0) + signal_peptide.count(aa)
        comp = []
        for key in training_composition:
            if training_composition[key] == 0:
                comp.append(0)
            else:
                comp.append(round((training_composition[key] / length), 3))

        pa = ProteinAnalysis(signal_peptide)
        hp = pa.protein_scale(hphob, 5)
        # comp.append(scale(max(hp)))
        # comp.append(scale((sum(hp) / length)))
        vector.append(comp)

    return np.array(vector)


non_std = ['X', 'U', 'O', 'J', 'Z', 'B']


def get_features_class(df, length):
    vector = []
    seq_class = []
    hphob = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
             'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
             'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
             'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
             'X': 0}
    for j, row in df.iterrows():
        sequence = str(row["Sequence"])
        tot = sequence.count('X') + sequence.count('U') + sequence.count('O') + sequence.count('J') + sequence.count('Z') + sequence.count('B')
        if tot == 0:
            if pd.isnull(row["End"]) is True:
                seq_class.append(0)
            else:
                seq_class.append(1)
            signal_peptide = sequence[:length]
            training_composition = {}
            for aa in amino:
                training_composition[aa] = training_composition.get(aa, 0) + signal_peptide.count(aa)
            comp = []
            for key in training_composition:
                comp.append(round((training_composition[key] / length), 3))
            prot_ends = 'XX' + signal_peptide + 'XX'
            pa = ProteinAnalysis(prot_ends)
            hp = pa.protein_scale(hphob, 5)
            comp.append(scale(max(hp)))
            comp.append(scale((sum(hp) / length)))
            vector.append(comp)
    return np.array(vector), np.array(seq_class)


def get_features1(df, length):
    vector = []
    for j, row in df.iterrows():
        sequence = str(row["Sequence"])
        tot = sequence.count('X') + sequence.count('U') + sequence.count('O') + sequence.count('J') + sequence.count('Z') + sequence.count('B')
        if tot == 0:
            signal_peptide = sequence[:length]
            training_composition = {}
            for aa in amino:
                training_composition[aa] = training_composition.get(aa, 0) + signal_peptide.count(aa)
            comp = []
            for key in training_composition:
                comp.append((training_composition[key] / length))
            prot_ends = 'XXX' + signal_peptide + 'XXX'
            pa = ProteinAnalysis(prot_ends)
            hp = pa.protein_scale(hphob, 7)
            # comp.append(scale(max(hp)))
            # comp.append(scale((sum(hp) / length)))
            max_hp = max(hp)
            comp.append(max_hp)
            comp.append((sum(hp) / length))
            comp.append((hp.index(max_hp) / length))
            ah = pa.protein_scale(ahelix, 7)
            comp.append((sum(ah) / length))
            # comp.append((ah.index(max(ah)) / length))
            vector.append(comp)

    f_vector = np.array(vector)
    # columns_to_scale = [20, 21, 22, 23, 24]
    # for column in columns_to_scale:
    #     column_values = f_vector[:, column]
    #     min_value = column_values.min()
    #     max_value = column_values.max()
    #     scaled_column = (column_values - min_value) / (max_value - min_value)
    #     f_vector[:, column] = scaled_column
    return pre.MinMaxScaler().fit_transform(f_vector)
    # return f_vector