import sapiens
import sys
import pandas as pd
import numpy as np
import pickle as pkl
import os
from tqdm import tqdm

sys.path.append(os.getcwd()+"/src")
from utils import get_pseudo_likelihood
class Sapiens():

    """
    Class for the protein Model Sapiens
    Author: Aurora
    """

    def __init__(self, chain_type="H", method="average", file_name = "."):
        """
        Creates the instance of the language model instance

        parameters
        ----------

        chain_type: `str`
        `L` or `H` whether the input is from light or heavy chains resprectively
        
        method: `str`
        Layer that we want the embedings from

        file_name: `str`
        The name of the folder to store the embeddings
        """

        self.chain = chain_type
        if isinstance (method,int):
            self.layer = method
        elif method == "average":
            self.layer = None
        else:
            self.layer = "prob"
        self.file = file_name

    def fit_transform(self, sequences, starts, ends):
        """
        Fits the model and outputs the embeddings.
        
        parameters
        ----------

        sequences: `list` 
        Column with sequences to be transformed
        ------

        None, saved the embeddings in the embeddings.csv
        """

        if self.layer == None:
            print("Using the average layer")
            output = []
            for j,sequence in enumerate(sequences):
               output.append(list(np.mean(np.mean(sapiens.predict_residue_embedding(sequence, chain_type=self.chain)[:,starts[j]:ends[j],:], axis = 1),axis = 0)))
            # output = sequences.apply(lambda seq: pd.Series(np.mean(np.mean(sapiens.predict_residue_embedding(seq, chain_type=self.chain)[:,starts[seq.name]:ends[seq.name],:], axis = 1),axis = 0)))
            # output.to_csv("outfiles/"+self.file+"/embeddings.csv") #We have one embeded sequence per row
            output = pd.DataFrame(output, columns=[f"dim_{i}" for i in range(len(output[0]))])
            return output.reset_index(drop=True)
        elif self.layer == "prob":
            print("\n Making probabilities")
            output = sequences.apply(lambda seq: pd.DataFrame(sapiens.predict_scores(seq, chain_type=self.chain)))
            embedings = get_pseudo_likelihood(output, sequences)
            pkl.dump([output,embedings],open("outfiles/"+self.file+"/probabilities_pseudo.pkl","wb"))
        else:
            print("\nUsing the {} layer".format(self.layer))
            output = sequences.apply(lambda seq: pd.Series(sapiens.predict_sequence_embedding(seq, chain_type=self.chain, layer=self.layer)))
            # output.to_csv("outfiles/"+self.file+"/embeddings.csv") #We have one embeded sequence per row
            output.columns = [f"dim_{i}" for i in range(output.shape[1])]
            return output.reset_index(drop=True)

    def calc_pseudo_likelihood_sequence(self, sequences:list,starts, ends):
        
       	likelihoods_info = []
        pll_all_sequences = []
        self.mask_model = self.mask_model.to(self.device)

        for j,sequence in enumerate(tqdm(sequences)):
            try: 
                amino_acids = list(sequence)
                seq_tokens = ' '.join(amino_acids)
                seq_tokens = self.tokenizer(seq_tokens, return_tensors='pt')
                seq_tokens = seq_tokens.to(self.device)
                logits = self.mask_model(**seq_tokens).logits[0].cpu().detach().numpy()
                prob = scipy.special.softmax(logits,axis = 1)
                df = pd.DataFrame(prob, columns = self.tokenizer.convert_ids_to_tokens(range(0,33)))
                df = df.iloc[1:-1,:]

                per_position_ll = []
                for i in range(starts[j],ends[j]):
                    aa_i = amino_acids[i]
                    ll_i = np.log(df.iloc[i,:][aa_i])
                    per_position_ll.append(ll_i)
                
               	pll_seq = np.average(per_position_ll)
                pll_all_sequences.append(pll_seq)
                likelihoods_info.append({"Sequence": sequence, "Likelihood": pll_seq})
            except:
                likelihoods_info.append({"Sequence": sequence, "Likelihood": None})

        result_df = pd.DataFrame(likelihoods_info)
        return result_df    
    
    def calc_probability_matrix(self, sequence:str):
        df = pd.DataFrame(sapiens.predict_scores(sequence, chain_type=self.chain))
        return df

    def generate_probability_matrix_csv(self, sequences):
        """
        Generates and saves the probability matrix CSV file.
        """
        prob_matrix = []
        best_sequences = []
        for i, seq in enumerate(sequences):
            per_pos = self.calc_probability_matrix(sequence=seq)
            prob_matrix.append(per_pos)
            best_sequence = "".join(per_pos.idxmax(axis=1))
            best_sequences.append(best_sequence)
            if i < len(sequences) - 1:
                separator_row = pd.DataFrame(index=[f"Sequence {i + 1} separator"], columns=per_pos.columns)
                prob_matrix.append(separator_row)
        combined_matrix = pd.concat(prob_matrix)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Directory '{output_dir}' created.")

    # Save concatenated probabilities to CSV
        csv_path = os.path.join(output_dir, "probabilities_pseudo_sapiens.csv")
        combined_matrix.to_csv(csv_path, index=False)
        print(f"Saved probabilities to {csv_path}")
        print(best_sequences)
        print("This is saved")
        self.best_sequences = best_sequences
        
    def best_sequences(self,sequences:list,starts,ends):
        best_sequences = self.best_sequences
        best_starts = [0] * len(best_sequences)
        best_ends = [len(seq) for seq in best_sequences]
   
        starts = best_starts
        ends = best_ends

        pseudo_likelihoods = self.calc_pseudo_likelihood_sequence(best_sequences, starts, ends)
        print(sequences,likelihoods,best_sequences,pseudo_likelihoods)
    # Ensure all arrays have the same length
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

