# evo-research

# Setting up the environment

Create a new environment using Python3.8 as: conda create -n likelihoods python=3.8

Activate the environment: conda activate likelihoods

To install necessary libraries: pip install -r requirements.txt

# To run the pipeline


To create embeddings, calculate likelihoods and get n mutations you need to run the embeddings_calculator.py function:

--dataset Specify the name of your desired dataset that you want to analyze

--mode For now it is only "general" for general protein analysis

--file_path Specify the path of the dataset file

--num_mutations Specify the number of mutations you would like to make to the sequences (iteratively)


To create evolocities:

-Import first the evolocity package

-f Specify the file path to your EMBEDDINGS that are created beforehand

-m Specify the model you would like to use for the evo-velocity mapping, currently there are ESM, ablang, protbert, sapiens. It is recommended to use the same model for the same embeddings.
