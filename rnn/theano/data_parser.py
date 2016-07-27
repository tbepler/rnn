from Bio import SeqIO
from Bio import SwissProt
import numpy as np
import datetime
import re
import collections
import urllib2

def import_seq_data(uniprot_file, pdb_file):
    # Imports and preprocesses the sequence data
    # Takes file and returns sequence and labels of the sequence in np array format
    sequence_array = []
    label_array = []
    seq_dict = {'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9, 'M':10, 'N':11, 'P':12, 'Q':13, 'R':14, 'S':15, 'T':16, 'V':17, 'W':18, 'Y':19}
    label_dict = {"HELIX": 1, "STRAND": 2, "TURN":3}
    pdb_label_dict = {" ": 0, "H": 1, "B": 2, "E":3, "G": 4, "I":5, "T": 6, "S": 7}
    # H = alpha helix
    # B = residue in isolated beta-bridge
    # E = extended strand, participates in beta ladder
    # G = 3-helix (3/10 helix)
    # I = 5 helix (pi helix)
    # T = hydrogen bonded turn
    # S = bend
    num_labels = len(label_dict) + 1
    max_len = 0
    final_frequencies = [0] * num_labels

    # Make a dictonary of pdb protein sequence + structures keyed by name of sequence
    pdb_seqs = {}
    with open(pdb_file, 'r') as myfile:
        pdb_string = myfile.read().replace('\n', '')
    pdb_string = pdb_string.split(">")
    pdb_string = [re.split(":secstr|:sequence", strsplit) for strsplit in pdb_string]
    pdb_string = pdb_string[1:]
    for record in pdb_string:
        if record[0] not in pdb_seqs:
            # Make AA sequence for pdb
            seq = []
            for seq_label in record[1]:
                if seq == None:
                    continue
                if seq_label in seq_dict:
                    seq += [seq_dict[seq_label]]
                else:
                    seq = None
            pdb_seqs[record[0]] = [seq]
        else:
            # Make secondary structure sequnence for pdb
            ss = []
            for ss_label in record[1]:
                ss += [pdb_label_dict]
            pdb_seqs[record[0]] += [ss]
    consolidated_pdb = {}
    for seq in pdb_seqs.keys():
        short = seq.split(":")[0]
        if short in consolidated_pdb:
            consolidated_pdb[short] += [pdb_seqs[seq]]
        else:
            consolidated_pdb[short] = [pdb_seqs[seq]]
    pdb_names = consolidated_pdb.keys()
    #string = ""
    #for i in string_names:
    #    string += i + " "
    #text_file = open("pdb_string.txt", "w")
    #text_file.write(string)
    #text_file.close()

    url_template = "http://www.rcsb.org/pdb/files/{}.pdb"
    name_table = {}
    counter = 0
    for name in pdb_names:
        counter += 1
        #print name
        if consolidated_pdb[name][0] == None:
            continue
        try:
            url = url_template.format(name)
            response = urllib2.urlopen(url)
            pdb = response.read()
            response.close()  # best practice to close the file
            m = re.search('UNP\ +(\w+)', pdb)
        except urllib2.HTTPError as err:
            print err
            print name
        if m == None:
            continue
        unp = m.group(1)
        #print unp
        name_table[unp] = name
        if counter % 100 == 0:
            #np.save("predata", [name_table, consolidated_pdb])
            print "done"
    np.save("predata", [name_table, consolidated_pdb])
    print "done done"

    # gets next record from the parsed file from SwissProt
    for record in SwissProt.parse(open(uniprot_file)):
        print record.entry_name
        labels = [0]*record.sequence_length
        frequencies = []
        changed = False
        if record.entry_name in name_table:
            for feature in record.features:
                # for each feature, if it is in the label dictionary, add corresponding label to array and update label frequencies
                if feature[0] in label_dict:
                    new_label = label_dict[feature[0]]
                else:
                    continue
                begin = feature[1]
                end = feature[2]
                # there is a label in the sequence and adds these labels to the full array
                changed = True
                for i in range(begin-1, end-1):
                    labels[i] = new_label
            encoded_seq = []
            if not changed:
                continue
            counter = dict(collections.Counter(labels))
            frequencies = [counter[i] if i in counter else 0 for i in range(num_labels)]
            final_frequencies = map(add, final_frequencies, frequencies)
            label_array.append(labels) 
            sequence_array.append(encoded_seq)
            if record.sequence_length > max_len:
                max_len = record.sequence_length
            for i in range(pdb_seqs[name_table[record.entry_name]]):
                sequence_array.append(pdb_seqs[name_table[record.entry_name]][i][0])
                label_array.append([pdb_seqs[name_table[record.entry_name]][i][1], labels])

    return [sequence_array, label_array, num_labels, final_frequencies]

if __name__ == '__main__':
    currentTime = datetime.datetime.now() 
    startTime = currentTime
    output = import_seq_data("uniprot_all.txt", "ss.txt")
    print output
    np.save("predata", output)

