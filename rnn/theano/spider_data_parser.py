from Bio import SeqIO
from Bio import SwissProt
import numpy as np
import datetime
import re
import collections
import urllib2
from operator import add

def import_seq_data(spider_file):
    # Imports and preprocesses the sequence data
    # Takes file and returns sequence and labels of the sequence in np array format
    sequence_array = []
    label_array = []
    seq_dict = {'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9, 'M':10, 'N':11, 'P':12, 'Q':13, 'R':14, 'S':15, 'T':16, 'V':17, 'W':18, 'Y':19}
    label_dict = {"X": 0, "H": 1, "C": 2, "E":3}
    # H = alpha helix
    # B = residue in isolated beta-bridge
    # E = extended strand, participates in beta ladder
    num_labels = len(label_dict)
    final_frequencies = [0] * num_labels
    seq_len = 0

    # Make a dictonary of protein sequence + structures keyed by name of sequence
    fasta_dict = SeqIO.index(spider_file, "fasta")
    seqstruct_dict = {}
    for key in fasta_dict.keys():
        seqstruct = fasta_dict.get_raw(key).split("\n")[1:3]
        
        # Make AA sequence 
        seq = []
        for seq_label in seqstruct[0]:
            if seq_label in seq_dict:
                seq += [seq_dict[seq_label]]
            else:
                raise ValueError('Amino acid label not in dict')
        seqstruct[0] = seq
        # Make SS sequence 
        ss = []
        for ss_label in seqstruct[1]:
            if ss_label in label_dict:
                ss += [label_dict[ss_label]]
            else:
                raise ValueError('Secondary structure label not in dict')
        seqstruct[1] = ss 
        seqstruct_dict[key] = seqstruct

    # Construct the seq, label arrays and frequencies of labels
    for value in seqstruct_dict.values():
        sequence_array += [value[0]]
        labels = value[1]
        if seq_len < len(labels):
            seq_len = len(labels)
        counter = dict(collections.Counter(labels))
        frequencies = [counter[i] if i in counter else 0 for i in range(num_labels)]
        final_frequencies = map(add, final_frequencies, frequencies)
        label_array += [labels]

    return [sequence_array, label_array, num_labels, final_frequencies]

if __name__ == '__main__':
    currentTime = datetime.datetime.now() 
    startTime = currentTime
    output = import_seq_data("/n/scratch2/rkc10/seq+ss_train.txt")
    np.save("/n/scratch2/rkc10/train_predata", output)
    test_output = import_seq_data("/n/scratch2/rkc10/seq+ss_test1199.txt")
    np.save("/n/scratch2/rkc10/test_predata", test_output)
