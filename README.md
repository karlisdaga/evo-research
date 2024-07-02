# evo-research

# Setting up the environment

Create a new environment using Python3.8 as: conda create -n likelihoods python=3.8

Activate the environment: conda activate likelihoods

To install necessary libraries: pip install -r requirements.txt

# Pipeline

To successfully run the pipeline, the input can be a csv, tsv or any other format as long as the delimiter in the code is set correctly.
--dataset Specify the name of your desired dataset that you want to analyze
--file_path Specify the path of the dataset file
--column Specify the column that the pipeline reads, for example column = "full_sequence"

In the code init_list can be changed to how many models you would like to test. Default is all four. Please keep in mind their specific notation.

OPTIONAL:
--num_mutations Specify how many preffered mutations in a sequence needs to be done. This function will return a sequence which iteratively chooses the top substituted amino acid predicted by the specified PLM.


FOR EMBEDDINGS:

To create a folder containing embeddings from each model the fit_transform function is used

Provides with the specific amount of dimension embeddings for the sequence. 
Layer can be changed from first to average to last in the PLM_models directory

FOR PROBABILITIES: 

To create a separate folder called "probabilities":
In case of Sapiens: generate_probability_matrix_csv
For all other models : calc_evo_likelihood_matrix_per_position

Sequences are separated via an empty row
Each position provides with probabilities for all 20 amino acids

FOR LIKELIHOODS:

This will return a folder where also embeddings are stored and create these likelihood datasets f"evo_likelihood_{suffixes[i]}.csv"

For all the models to create likelihoods this is the following function:
calc_pseudo_likelihood_sequence

The dataset returned will have the sequence and likelihood column

FOR PREDICTED MUTATIONS:

For top n-predicted mutations the following function is used:
n_mut_sequences and then following n_mut_sequences.to_csv

This returns a dataset of original sequence, its evolutionary likelihood, mutated sequence and its likelihood

N-Mutations are made by changing the top amino acid substitutions from the original sequence

# Evolocities

Import first the evolocity package/ pip install evolocity

--f Specify the file path to your EMBEDDINGS that are created beforehand

--m Specify the model you would like to use for the evo-velocity mapping, currently there are ESM, ablang, protbert, sapiens. It is recommended to use the same model for the same embeddings.
