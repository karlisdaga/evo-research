import torch
from evolocity.tools.fb_model import FBModel
import argparse
import pandas as pd
import anndata
import evolocity as evo
import scanpy as sc

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--dataset_file', help='Path to the dataset file containing embeddings')
args = parser.parse_args()

dataset_file = args.dataset_file
model_name = "esm1_t6_43M_UR50S"
model = FBModel(name=model_name, repr_layer=[-1], random_init=False)
# Load dataset file (calculated embeddings)
embeddings_df = pd.read_csv(dataset_file)
embeddings_df = embeddings_df.dropna(subset=['full_sequence'])
embeddings_df['full_sequence'] = embeddings_df['full_sequence'].astype(str)
# Extract sequences and other metadata from the dataset(specified as strings)
sequences = embeddings_df['full_sequence']
sequences = [str(seq) for seq in sequences]
vdj_cgene = [str(gene) for gene in embeddings_df['VDJ_cgene']]
sample_ids =[str(sample_id) for sample_id in embeddings_df['sample_id']]
embedding_cols = [col for col in embeddings_df.columns if col.startswith('dim')]
metadata_cols = [col for col in embeddings_df.columns if col not in embedding_cols]

# Create an AnnData object
adata = anndata.AnnData(embeddings_df[embedding_cols])

# add sequence and metadata to .obs slot of AnnData object (for now VDJ gene sequences)
adata.obs['seq'] = sequences
adata.obs['VDJ_cgene'] = vdj_cgene
adata.obs['sample_id'] = sample_ids

#scanpy it as tsn( Choices are: pca, tsne, umap)
sc.tl.tsne(adata)
# Construct sequence similarity network
evo.pp.neighbors(adata)
basis = "tsne"
# Run evolocity analysis
evo.tl.velocity_graph(adata)

# Embed network and velocities in two-dimensions
evo.tl.velocity_embedding(adata, basis = basis)
#if model appears in the data
if 'model' in adata.uns:
    del adata.uns['model']
# Save the processed AnnData object to a file
output_file = "processed_data.h5ad"  # Adjust the filename and format as needed
adata.write(output_file)

print(f"Processed data saved to {output_file}")
