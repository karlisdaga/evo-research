import pandas as pd
import numpy as np
import os
import sys
import argparse

sys.path.append("/hpc/dla_lti/kdagakrumins/anaconda3/PLM_anamay/src/")

from ablang_model import Ablang
from ESM_model import ESM
from sapiens_model import Sapiens
from protbert import ProtBert
parser = argparse.ArgumentParser()

parser.add_argument('-d','--dataset',required=False)   
parser.add_argument('--mode')
parser.add_argument('--file_path', required=False)
parser.add_argument('--num_mutations', type=int)
parser.add_argument("--column")

args = parser.parse_args()

dataset = args.dataset
mode = args.mode
num_mutations = args.num_mutations
column = args.column

init_list = [ESM, Sapiens, Ablang, ProtBert]
suffixes = ["ESM", "sapiens", "ablang", "protbert"]

if mode == "general":
 
    repertoire_file  = pd.read_csv(args.file_path, delimiter=",")
    repertoire_file = repertoire_file.dropna(subset=[column])
    repertoire_file[column] = repertoire_file[column].str.replace("_", "")
    repertoire_file_folder = os.path.dirname(args.file_path)
    save_path = os.path.join(repertoire_file_folder,"Variant_tree2_LC_test")

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    starts = [0]*repertoire_file.shape[0]
    ends = repertoire_file[column].apply(len)

    for i,model in enumerate(init_list): ### the class method "fit_transform" calculates the embedding of each sequence
     
        # if os.path.exists(os.path.join(save_path,f"embeddings_{suffixes[i]}.csv")):
        #     continue

        if suffixes[i] == "ablang":
            model = model(chain = "light")
        elif suffixes[i] == "sapiens":
            model = model(chain_type = "L")
            sequences = repertoire_file[column]
            per_pos = model.generate_probability_matrix_csv(sequences)
            evo_lhood = model.calc_pseudo_likelihood_sequence(sequences=list(repertoire_file[column]),starts=list(starts),ends=list(ends))
            evo_lhood.to_csv(os.path.join(save_path,f"evo_likelihood_sapiens"), index = False)
          #  n_mut_sequences = model.mutate_sequences(num_mutations)
          #  n_mut_sequences.to_csv(os.path.join(save_path,f"n_{num_mutations}_mut_sequences_sapiens.csv"), index = False)
        else:
            model = model()

        embeds = model.fit_transform(sequences=list(repertoire_file[column]),starts=list(starts),ends=list(ends))
        embeds = pd.concat([repertoire_file,embeds],axis=1)
        embeds.to_csv(os.path.join(save_path,f"embeddings_{suffixes[i]}.csv"), index=False)
        if suffixes[i] == "ablang" or suffixes[i] == "ESM" or suffixes[i] == "protbert":
            per_pos = model.calc_evo_likelihood_matrix_per_position(sequences=list(repertoire_file[column]))
            evo_lhood = model.calc_pseudo_likelihood_sequence(sequences=list(repertoire_file[column]),starts=list(starts),ends=list(ends))
            evo_lhood.to_csv(os.path.join(save_path,f"evo_likelihood_{suffixes[i]}.csv"), index = False)
#            n_mut_sequences = model.mutate_sequences(num_mutations)
#            n_mut_sequences.to_csv(os.path.join(save_path,f"n_{num_mutations}_mut_sequences_{suffixes[i]}.csv"), index = False)
#            delta_likelihood = model.delta_likelihood(sequences=list(repertoire_file["Mutated_sequence"]),starts=list(starts),ends=list(ends),num_mutations=num_mutations)
#            delta_likelihood.to_csv(os.path.join(save_path,f"n_{num_mutations}_delta_likelihood_{suffixes[i]}.csv"), index = False)

