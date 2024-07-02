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

FOR PROBABILITIES: 

To create a separate folder called "probabilities":
In case of Sapiens: per_pos = model.generate_probability_matrix_csv(sequences)
For all other models : per_pos = model.calc_evo_likelihood_matrix_per_position(sequences=list(repertoire_file[column]))

FOR LIKELIHOODS:

This will return a folder where also embeddings are stored and create these likelihood f"evo_likelihood_{suffixes[i]}.csv"

For all the models to create likelihoods this is the following function:
calc_pseudo_likelihood_sequence

To create evolocities:

-Import first the evolocity package

-f Specify the file path to your EMBEDDINGS that are created beforehand

-m Specify the model you would like to use for the evo-velocity mapping, currently there are ESM, ablang, protbert, sapiens. It is recommended to use the same model for the same embeddings.
