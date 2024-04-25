import ablang
import numpy as np
import pandas as pd
import pickle as pkl
import scipy
from tqdm import tqdm
import os
import sys
import torch

sys.path.append("../scripts")

from utils import get_pseudo_likelihood
class Ablang():

    """
    Class for the protein Model Ablang
    """

    def __init__(self, chain = "heavy",file_name = ".", method = "seqcoding"):
        """
        Creates the instance of the language model instance; either light or heavy
        
        method: `str`
        Which token to use to extract embeddings of the last layer
        
        file_name: `str`
        The name of the folder to store the embeddings
        """
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = ablang.pretrained(chain,device=self.device)
        #dont update the weights
        self.model.freeze()

        self.file = file_name
        self.mode = method



    def fit_transform(self, sequences, starts, ends):
        """
        Fits the model and outputs the embeddings.
        
        parameters
        ----------

        sequences: `list` 
        List with sequences to be transformed
        ------

        None, saved the embeddings in the embeddings.csv
        """
        output = self.model(sequences, mode=self.mode)
        if self.mode == "seqcoding":
            #The embeddings are made my averaging across all residues    
            # pd.DataFrame(output).to_csv("outfiles/"+self.file+"/embeddings.csv")
            return pd.DataFrame(output,columns=[f"dim_{i}" for i in range(output.shape[1])])

    def calc_evo_likelihood_matrix_per_position(self, sequences:list):

        probs = []
        for sequence in tqdm(sequences):
            logits = self.model(sequence, mode="likelihood")[0]
            prob = scipy.special.softmax(logits,axis = 1)
            df = pd.DataFrame(prob, columns = list(self.model.tokenizer.vocab_to_aa.values())[4:])
            #removing CLS and SEP
            df = df.iloc[1:-1,:]
            df = df.reindex(sorted(df.columns), axis=1)
            probs.append(df)

        likelihoods = get_pseudo_likelihood(probs, sequences)
        probs_concatenated = pd.DataFrame()

    # Iterate over each sequence and its corresponding DataFrame
        for i, df in enumerate(probs):
        # Add a blank row as a separator between sequences
            separator_row = pd.DataFrame(index=[f"Sequence {i + 1}"], columns=df.columns)
            probs_concatenated = pd.concat([probs_concatenated, separator_row, df])

        probs_concatenated.reset_index(drop=True, inplace=True)
        prob_by_column = {}
        for column in probs_concatenated.columns:
           prob_by_column[column] = probs_concatenated[column]

        prob_by_column_concatenated = pd.concat(prob_by_column, axis=1)
        output_dir = "probabilities"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Directory '{output_dir}' created.")

    # Save concatenated probabilities to CSV
        csv_path = os.path.join(output_dir, "probabilities_pseudo_ablang.csv")
        probs_concatenated.to_csv(csv_path, index=False)
        print(f"Saved probabilities to {csv_path}")


        best_sequences = []

        for i, df in enumerate(probs):
            best_sequence = ""
            for _, row in df.iterrows():
        # Find the amino acid with the highest probability
                best_amino_acid = row.idxmax()
                best_amino_acid = str(best_amino_acid)
                best_sequence += best_amino_acid
            best_sequences.append(best_sequence)

        self.best_sequences = best_sequences


    def calc_pseudo_likelihood_sequence(self, sequences:list,starts, ends):
        pll_all_sequences = []
        for j,sequence in enumerate(tqdm(sequences)):
            try:
                amino_acids = list(sequence)
                logits = self.model(sequence, mode="likelihood")[0]
                prob = scipy.special.softmax(logits,axis = 1)
                df = pd.DataFrame(prob, columns = list(self.model.tokenizer.vocab_to_aa.values())[4:])
                df = df.iloc[1:-1,:]
                df = df.reindex(sorted(df.columns), axis=1)


                per_position_ll = []
                for i in range(starts[j],ends[j]):
                    aa_i = amino_acids[i]
                    if aa_i == "-" or aa_i == "*":
                        continue
                    ll_i = np.log(df.iloc[i,:][aa_i])
                    per_position_ll.append(ll_i)
            

                pll_seq = np.average(per_position_ll)
                pll_all_sequences.append(pll_seq)
            except:
                pll_all_sequences.append(None)

        return pll_all_sequences    

    def calc_probability_matrix(self, sequence:str):
        logits = self.model(sequence, mode="likelihood")[0]
        prob = scipy.special.softmax(logits,axis = 1)
        df = pd.DataFrame(prob, columns = list(self.model.tokenizer.vocab_to_aa.values())[4:])
        df = df.iloc[1:-1,:]
        df = df.reindex(sorted(df.columns), axis=1)

        return df

    def process_sequences(self, sequences: list, starts,ends):
    # Calculate evolutionary likelihoods for each sequence
            likelihoods = self.calc_pseudo_likelihood_sequence(sequences, starts, ends)
            sequences_processed = sequences
            result_df = pd.DataFrame({"Sequence": sequences_processed, "Likelihood": likelihoods})
            return result_df
    def best_sequences(self,sequences:list,starts,ends):
        best_sequences = self.best_sequences
        best_starts = [0] * len(best_sequences)
        best_ends = [len(seq) for seq in best_sequences]
   
   
        starts = best_starts
        ends = best_ends

        pseudo_likelihoods = self.calc_pseudo_likelihood_sequence(best_sequences, starts, ends)
 

        if len(sequences) != len(starts) or len(sequences) != len(ends):
            raise ValueError("Lengths of sequences, starts, and ends must be equal.")

        df_result = pd.DataFrame(columns=['Original_sequence', 'Evo_likelihood_original', 'Best_sequence', 'Pseudo_likelihood_best'])


        for i, (seq, likelihood, best_seq, pseudo_likelihood) in enumerate(zip(sequences, likelihoods, best_sequences, pseudo_likelihoods)):
            df_result.loc[i] = [seq, likelihood, best_seq, pseudo_likelihood]
        self.df_result = df_result
        return df_result

    def mutate_sequences(self, num_mutations):
        mutated_sequences = []

        for i,row in self.df_result.iterrows():
            best_seq = row["Best_sequence"]
            original_seq = row["Original_sequence"]
            count = 0
            mutated_sequence = ""

            for best_aa, original_aa in zip(best_seq, original_seq):
                if count < num_mutations:
                    if best_aa == original_aa:
                        mutated_sequence += original_aa
                    else:
                        mutated_sequence += best_aa
                        count += 1
                else:
                    mutated_sequence += original_seq[len(mutated_sequence):]
                    break
            mutated_sequences.append(mutated_sequence)

        df_mutated_sequences = pd.DataFrame({'Mutated_sequence': mutated_sequences})

        print(df_mutated_sequences)
        return df_mutated_sequences

