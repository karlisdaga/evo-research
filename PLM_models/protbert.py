from transformers import BertModel, BertTokenizer, BertForMaskedLM
import pandas as pd
import numpy as np
from tqdm import tqdm 
import torch
import scipy
import os
import sys
from utils import get_pseudo_likelihood
import pickle as pkl

sys.path.append("../scripts")
class ProtBert():

    """
    Class for the protein Language Model
    """

    def __init__(self, method = "average", file_name = "."):
        """
        Creates the instance of the language model instance, loads tokenizer and model

        parameters
        ----------

        method: `str`
        Which token to use to extract embeddings of the last layer
        
        file_name: `str`
        The name of the folder to store the embeddings
        """
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

        self.model = BertModel.from_pretrained("Rostlab/prot_bert").to(self.device)
    
        self.mask_model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert").to(self.device)

        self.method = method
        self.file = file_name


    def fit_transform(self, sequences:list, starts, ends, batches = 10):
        """
        Fits the model and outputs the embeddings.
        
        parameters
        ----------

        sequences: `list` 
        List with sequences to be transformed
        
        batches: `int`
        Number of batches. Per batch a checkpoint file will be saved

        return
        ------

        None, saved the embeddings in the embeddings.csv
        """
        batch_size = round(len(sequences)/batches)
        
        pooler_zero = np.zeros((len(sequences), 1024))
        print("\nUsing the {} method".format(self.method))
        for sequence,_ in zip(enumerate(sequences), tqdm(range(len(sequences)))):
            if not isinstance(sequence[1], float):
                j = sequence[0]
                seq_tokens = ' '.join(list(sequence[1]))
                tokenized_sequences = self.tokenizer(seq_tokens, return_tensors= 'pt') #return tensors using pytorch
                tokenized_sequences = tokenized_sequences.to(self.device)
                output = self.model(**tokenized_sequences)
                
                if self.method == "average":
                    output = torch.mean(output.last_hidden_state[:,starts[j]:ends[j],:], axis = 1)[0]
                
                elif self.method == "pooler":
                    output = output.pooler_output[0]
                elif self.method == "first":
                    output = output.last_hidden_state[:,starts[j]][0]
                elif self.method == "last":
                    output = output.last_hidden_state[:,ends[j]-1][0]
                    
                pooler_zero[sequence[0],:] = output.tolist()
                # if sequence[0] % (batch_size+1) == 0:   #Checkpoint save
                #     pd.DataFrame(pooler_zero).to_csv("outfiles/"+self.file+"/embeddings.csv") 

        # pd.DataFrame(pooler_zero).to_csv("outfiles/"+self.file+"/embeddings.csv")
        
        return pd.DataFrame(pooler_zero,columns=[f"dim_{i}" for i in range(pooler_zero.shape[1])])

    def calc_evo_likelihood_matrix_per_position(self, sequences: list):
        probs = []
        self.mask_model = self.mask_model.to(self.device)

        for sequence in tqdm(sequences):
            seq_tokens = ' '.join(list(sequence))
            seq_tokens = self.tokenizer(seq_tokens, return_tensors='pt')
            seq_tokens = seq_tokens.to(self.device)
            logits = self.mask_model(**seq_tokens).logits[0].cpu().detach().numpy()
            prob = scipy.special.softmax(logits,axis = 1)
            df = pd.DataFrame(prob, columns = self.tokenizer.vocab)
            df = df.iloc[:,5:-5]
            df = df.loc[:, df.columns.isin(["U","Z","O","B","X"]) == False]
            #removing CLS and SEP
            df = df.iloc[1:-1,:]
            df = df.reindex(sorted(df.columns), axis=1)
            probs.append(df)

        likelihoods = get_pseudo_likelihood(probs, sequences)
        probs_concatenated = pd.DataFrame()


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
        csv_path = os.path.join(output_dir, "probabilities_pseudo_protbert.csv")
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
        amino_acids = list(sequence)
        seq_tokens = ' '.join(amino_acids)
        seq_tokens = self.tokenizer(seq_tokens, return_tensors='pt')
        seq_tokens = seq_tokens.to(self.device)
        logits = self.mask_model(**seq_tokens).logits[0].cpu().detach().numpy()
        prob = scipy.special.softmax(logits,axis = 1)
        df = pd.DataFrame(prob, columns = self.tokenizer.vocab)
        df = df.iloc[1:-1,:]

        return df

    def best_sequences(self,sequences:list,starts,ends):
        
        best_sequences = self.best_sequences
        best_starts = [0] * len(best_sequences)
        best_ends = [len(seq) for seq in best_sequences]
        
        starts = best_starts
        ends = best_ends

    # Calculate pseudo likelihoods for each best sequence
        pseudo_likelihoods = self.calc_pseudo_likelihood_sequence(best_sequences, starts, ends)

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

