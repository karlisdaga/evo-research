import pandas as pd
import numpy as np
import os
import sys
import argparse

sys.path.append("../PLM_models/")

from ablang_model import Ablang
from ESM_model import ESM
from sapiens_model import Sapiens
from protbert import ProtBert
parser = argparse.ArgumentParser( )

parser.add_argument('-d','--dataset',required=False)   
parser.add_argument('--mode') 
parser.add_argument('--file_path', required=False)
parser.add_argument('--num_mutations', type=int)


args = parser.parse_args()

dataset = args.dataset
mode = args.mode
num_mutations = args.num_mutations

init_list = [ESM, Sapiens, Ablang, ProtBert]
suffixes = ["ESM", "sapiens", "ablang", "protbert"]
#Make sure your sequence is named as full_sequence in the dataset, otherwise change it in the code
if mode == "general":
 
    repertoire_file  = pd.read_csv(args.file_path, delimiter=",")

    repertoire_file_folder = os.path.dirname(args.file_path)
    save_path = os.path.join(repertoire_file_folder,"embedding_data")

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    starts = [0]*repertoire_file.shape[0]
    ends = repertoire_file["full_sequence"].apply(len)

    for i,model in enumerate(init_list): ### the class method "fit_transform" calculates the embedding of each sequence
     
        # if os.path.exists(os.path.join(save_path,f"embeddings_{suffixes[i]}.csv")):
        #     continue

        if suffixes[i] == "ablang":
            model = model(chain = "heavy")
        elif suffixes[i] == "sapiens":
            model = model(chain_type = "H")
            sequences = repertoire_file["full_sequence"]
            per_pos = model.generate_probability_matrix_csv(sequences)
            best_evo = model.process_sequences(sequences=list(repertoire_file["full_sequence"]),starts=list(starts),ends=list(ends))
            best_evo.to_csv(os.path.join(save_path,f"best_sequence_evo_sapiens.csv"), index = False)
 #           n_mut_sequences = model.mutate_sequences(num_mutations)   ### n_mut_sequences gives an iterated mutated sequence based on n amount of mutations(has to run all the previous functions)
 #           n_mut_sequences.to_csv(os.path.join(save_path,f"n_{num_mutations}_mut_sequences_sapiens.csv"), index = False)
        else:
            model = model()

        embeds = model.fit_transform(sequences=list(repertoire_file["full_sequence"]),starts=list(starts),ends=list(ends))
        embeds = pd.concat([repertoire_file,embeds],axis=1)
        embeds.to_csv(os.path.join(save_path,f"embeddings_{suffixes[i]}.csv"), index=False)
        if suffixes[i] == "ablang" or suffixes[i] == "ESM" or suffixes[i] == "protbert":
            per_pos = model.calc_evo_likelihood_matrix_per_position(sequences=list(repertoire_file["full_sequence"]))
            best_evo = model.process_sequences(sequences=list(repertoire_file["full_sequence"]),starts=list(starts),ends=list(ends))
            best_evo.to_csv(os.path.join(save_path,f"best_sequence_evo_{suffixes[i]}.csv"), index = False)
#            n_mut_sequences = model.mutate_sequences(num_mutations)
#            n_mut_sequences.to_csv(os.path.join(save_path,f"n_{num_mutations}_mut_sequences_{suffixes[i]}.csv"), index = False)
