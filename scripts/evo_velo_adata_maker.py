import matplotlib
import torch
from evolocity.tools.fb_model import FBModel
import argparse
import pandas as pd
import anndata
import evolocity as evo
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import os
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--dataset_file', help='Path to the dataset file containing embeddings')
parser.add_argument("-m", "--model", help="Choose a model: ESM, ablang, protbert, sapiens")
args = parser.parse_args()
dataset_file = args.dataset_file
model = args.model

embeddings_df = pd.read_csv(dataset_file)
embeddings_df = embeddings_df.dropna(subset=['full_sequence'])
embeddings_df['full_sequence'] = embeddings_df['full_sequence'].astype(str)
#Sequences from embeddings and other metadata
sequences = embeddings_df['full_sequence']
sequences = [str(seq) for seq in sequences]
vdj_cgene = [str(gene) for gene in embeddings_df['j_gene']]
sample_ids =[str(sample_id) for sample_id in embeddings_df['sample_id']]
shm_count = [float(count) for count in embeddings_df["SHM_count"]]
embedding_cols = [col for col in embeddings_df.columns if col.startswith('dim')]
metadata_cols = [col for col in embeddings_df.columns if col not in embedding_cols]

#Create an AnnData object
adata = anndata.AnnData(embeddings_df[embedding_cols])

#Add sequence and metadata information to .obs slot of AnnData object
adata.obs['seq'] = sequences
adata.obs['j_gene'] = vdj_cgene
adata.obs['sample_id'] = sample_ids
adata.obs["SHM_count"] = shm_count

#Construct sequence similarity network
evo.pp.neighbors(adata)
sc.tl.umap(adata)
basis = "umap"
if model  == "ESM":
    evo.tl.velocity_graph(adata)
else:
    evo.tl.velocity_graph(adata, model_name = model)
if 'model' in adata.uns:
    del adata.uns['model']
#Embed network and velocities in two-dimensions
evo.tl.velocity_embedding(adata, basis = basis)
#Save the processed AnnData object to a file
output_file = "processed_data.h5ad" 
adata.write(output_file)
print(f"Processed data saved to {output_file}")
results_folder = "/../results_folder/"
# Plot your data using the color mapping (jgene or SHM_count), make sure data is categorical
color="SHM_count"
ax = evo.pl.velocity_embedding_stream(adata, color=color, legend_loc="right margin", show=False)
save_path = f"evo_velo_embedding_stream_{color}_{model}.png"
plt.savefig(save_path, bbox_inches="tight")
