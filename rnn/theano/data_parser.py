from Bio import SeqIO
from Bio import SwissProt
import numpy as np
import datetime
import re
import collections
import urllib2
from operator import add

def import_seq_data(uniprot_file, pdb_file):
    # Imports and preprocesses the sequence data
    # Takes file and returns sequence and labels of the sequence in np array format
    sequence_array = []
    label_array = []
    seq_dict = {'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9, 'M':10, 'N':11, 'P':12, 'Q':13, 'R':14, 'S':15, 'T':16, 'V':17, 'W':18, 'Y':19}
    label_dict = {"SIGNAL": 0, "DNA_BING": 1, "BINDING": 2, "ZN_FING": 3, "TRANSMEM": 4, "INTRAMEM": 5, "ACT_SITE": 6, "NP_BIND": 7, "DOMAIN": 8}
    # not REGION, DOMAIN
    pdb_label_dict = {" ": 0, "H": 1, "B": 2, "E":3, "G": 4, "I":5, "T": 6, "S": 7}
    # H = alpha helix
    # B = residue in isolated beta-bridge
    # E = extended strand, participates in beta ladder
    # G = 3-helix (3/10 helix)
    # I = 5 helix (pi helix)
    # T = hydrogen bonded turn
    # S = bend
    num_labels = len(label_dict)
    max_len = 0
    final_frequencies = [0] * num_labels
    read_predata = True

    if not read_predata:
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
                    ss += [pdb_label_dict[ss_label]]
                pdb_seqs[record[0]] += [ss]
        # Dictonary of pdb sequences and labels keyed on the pdb name
        consolidated_pdb = {}
        for seq in pdb_seqs.keys():
            short = seq.split(":")[0]
            if pdb_seqs[seq][0] == None:
                continue
            if short in consolidated_pdb:
                consolidated_pdb[short] += [pdb_seqs[seq]]
            else:
                consolidated_pdb[short] = [pdb_seqs[seq]]
        # pdb names in the consolidated dictionary
        pdb_names = consolidated_pdb.keys()
        #string = ""
        #for i in string_names:
        #    string += i + " "
        #text_file = open("pdb_string.txt", "w")
        #text_file.write(string)
        #text_file.close()

        url_template = "http://www.rcsb.org/pdb/files/{}.pdb"
        # name_table: Dictionary of pdb names keyed by the uniprot name
        name_table = {}
        counter = 0
        for name in pdb_names:
            counter += 1
            #print name
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
        np.save("/n/scratch2/rkc10/predata", [name_table, consolidated_pdb])
        print "done done"
        np.load("~/nothing")
    else:
        name_table, consolidated_pdb = np.load("/n/scratch2/rkc10/predata.npy")
        #np.load("~/nothing")

    # gets next record from the parsed file from SwissProt
    swissprotdict = {}
    for record in SwissProt.parse(open(uniprot_file)):
        #print record.entry_name
        labels = [[0]*record.sequence_length for i in range(len(label_dict))]
        frequencies = []
        changed = False

        if record.accessions[0] in name_table: # TODO: Need to check if only one accession
            # Builds swissprot labels by looping through all features for a swissprot entry and matches it with the pdb entry
            for feature in record.features:
                # for each feature, if it is in the label dictionary, add corresponding label to array and update label frequencies
                print feature[0]
                if feature[0] not in swissprotdict:
                    swissprotdict[feature[0]] = 1
                else:
                    swissprotdict[feature[0]] += 1
                if feature[0] in label_dict:
                    new_label = label_dict[feature[0]]
                else:
                    continue
                begin = feature[1]
                end = feature[2]
                print begin
                print end
                if type(begin) is str:
                    print "Not int begin"
                    continue
                if type(end) is str:
                    try:
                        end = int(end.split(">")[-1])
                    except ValueError:
                        print "Not int end"
                        continue
                # there is a label in the sequence and adds these labels to the full array
                changed = True
                for i in range(begin, end-1):
                    labels[new_label][i] = 1
            # Make unp aa sequence
            encoded_seq = []
            for char in record.sequence:
                if char in seq_dict:
                    encoded_seq += [seq_dict[char]]
                else:
                    changed = False
            if len(encoded_seq) != record.sequence_length:
                print "Noooo way"
                changed = False
            if not changed:
                continue

            #counter = dict(collections.Counter(labels))
            #frequencies = [counter[i] if i in counter else 0 for i in range(num_labels)]
            frequencies = [sum(label) for label in labels]
            final_frequencies = map(add, final_frequencies, frequencies)
            if record.sequence_length > max_len:
                max_len = record.sequence_length

            pdb_array = consolidated_pdb[name_table[record.accessions[0]]]
            print "Next"
            for i in range(len(pdb_array)):
                print record.accessions[0]
                print name_table[record.accessions[0]]
                difference = len(pdb_array[i][0]) - len(labels[0])
                print "Difference: %s" % difference
                if difference == 0:
                    sequence_array.append(pdb_array[i][0])
                    full_labels = labels + [pdb_array[i][1]]
                    label_array.append(full_labels)

                elif difference < 0: # unp labels are longer and should find where pdb_array matches this
                    start = 0
                    end = 0
                    for j in range(-difference):
                        match = 0
                        for k in range(len(pdb_array[i][0])):
                            if encoded_seq[k+j] != pdb_array[i][0][k]:
                                continue
                            else:
                                match += 1
                                if match == len(pdb_array[i][0]):
                                    start = j
                                    end = j + k + 1
                    if start != 0 and end != 0:
                        print "Match found"
                        print pdb_array[i][0]
                        print encoded_seq[start:end]
                        sequence_array.append(pdb_array[i][0])
                        print "error here"
                        print labels
                        full_labels = [label[start:end] for label in labels] + [pdb_array[i][1]]
                        label_array.append(full_labels)
                    else:
                        print pdb_array[i][0]
                        print encoded_seq
                        print "Did not find match"

                elif difference > 0: # pdb labels longer
                    for j in range(difference):
                        match = 0
                        for k in range(len(labels[0])):
                            print len(encoded_seq)
                            print len(pdb_array[i][0])
                            if encoded_seq[k] != pdb_array[i][0][k+j]:
                                continue
                            else:
                                match += 1
                                if match == len(encoded_seq):
                                    start = j
                                    end = j + k + 1
                    if start != 0 and end != 0:
                        print "Match found"
                        print pdb_array[i][0][start:end]
                        print encoded_seq
                        sequence_array.append(pdb_array[i][0][start:end])
                        full_labels = labels + [pdb_array[i][1][start:end]]
                        label_array.append(full_labels)
                    else:
                        print pdb_array[i][0]
                        print encoded_seq
                        print "Did not find match"

    print "Number of pdb sequences: %s" % len(name_table)
    print "Number of unp sequences: %s" % len(pdb_array)
    print "Number of sequences: %s" % len(sequence_array)
    return [sequence_array, label_array, num_labels, final_frequencies]

if __name__ == '__main__':
    currentTime = datetime.datetime.now() 
    startTime = currentTime
    output = import_seq_data("/n/scratch2/rkc10/uniprot-all.txt", "/n/scratch2/rkc10/ss.txt")
    #print output
    np.save("/n/scratch2/rkc10/finaldata", output)

