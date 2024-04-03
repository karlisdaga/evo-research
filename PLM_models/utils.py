import os 
import pandas as pd
import numpy as np
from Bio.Align import PairwiseAligner

def prepare_sequence(path_to_file:str):
    """
    Prepare sequence strings of a given file, for langauge model by adding spaces

    parameters
    ---------

    path_to_file: `str`
    Path to file with sequences to be turned to embeddings


    return
    ------
    List of sequences with correct format
    """

    data = pd.read_csv(path_to_file)
    sequences = data["sequence"]
    print(sequences)
    sequences = sequences.apply(add_space)
    return sequences

def add_space(row):
    if not isinstance(row, float):
        row = " ".join(row)
    return row

def get_pseudo_likelihood(probs, sequences):
    probs_all = []
    for i,sequence_probs in enumerate(probs):
        wt_probs_full = []
        for pos in range(sequence_probs.shape[0]):

            wt_j = sequences[i][pos]
            #Can comment if PLM gives probabilities also for gaps
            if wt_j == "-" or wt_j == "*":
                continue
            wt_prob = sequence_probs.iloc[pos,:][wt_j]
            wt_probs_full.append(np.log(wt_prob))
        probs_all.append(np.average(wt_probs_full))
    return probs_all



def compute_evo_velocity(sequence_1, sequence_2, model):

    prob_mat_1 = model.calc_probability_matrix(sequence_1)
    prob_mat_2 = model.calc_probability_matrix(sequence_2)

    aligner = PairwiseAligner() 
    aligner.extend_gap_score = -0.1
    aligner.match_score = 5
    aligner.mismatch_score = -4
    aligner.open_gap_score = -4

    alignment = aligner.align(sequence_1,sequence_2)[0]
    alignment = alignment.aligned

    ranges_1 = alignment[0,:,:]
    ranges_2 = alignment[1,:,:]

    count = 0
    evo_velo = 0

    for i in range(ranges_1.shape[0]):
        start_1 = ranges_1[i,0]
        start_2 = ranges_2[i,0]
        subalign_len = ranges_1[i,1] - start_1

        for j in range(subalign_len):

            pos_1 = start_1 + j
            pos_2 = start_2 + j   

            amino_acid_1  = sequence_1[pos_1]
            amino_acid_2  = sequence_2[pos_2]

            if amino_acid_1 != amino_acid_2:

                evo_velo += (prob_mat_1[pos_1,amino_acid_2] - prob_mat_2[pos_2,amino_acid_1])
                count += 1

    evo_velo /= count

    return evo_velo 
