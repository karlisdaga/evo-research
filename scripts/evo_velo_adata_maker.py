import matplotlib
import torch
from evolocity.tools.fb_model import FBModel
import argparse
import pandas as pd
import anndata
import evolocity as evo
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib import cm
import numpy as np
import os
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--dataset_file', help='Path to the dataset file containing embeddings')
args = parser.parse_args()

dataset_file = args.dataset_file
model_name = "esm1_t6_43M_UR50S"

model = FBModel(name=model_name, repr_layer=[-1], random_init=False)

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
evo.tl.velocity_graph(adata)

#Embed network and velocities in two-dimensions
evo.tl.velocity_embedding(adata, basis = basis)
if 'model' in adata.uns:
    del adata.uns['model']

output_file = "evovelocity.h5ad" 
adata.write(output_file)
print(f"Processed data saved to {output_file}")
print(adata)
print(adata.obs.dtypes)
print(adata.obsm["velocity_umap"])
results_folder = "/hpc/dla_lti/kdagakrumins/anaconda3/PLM_anamay/results_folder"

print(adata.obs.dtypes)
print(adata.obs["SHM_count"].isnull().sum())
print(adata.obs["SHM_count"])
is_cat_shm_count = pd.api.types.is_categorical_dtype(adata.obs['SHM_count'])
print(f"Column 'SHM_count' is categorical: {is_cat_shm_count}")
# Plot your data using the color mapping (jgene or SHM_count)
color="SHM_count"
ax = evo.pl.velocity_embedding_stream(adata, color=color, legend_loc="right margin", show=False)
save_path = f"evo_velo_embedding_stream_{color}.png"
plt.savefig(save_path, bbox_inches="tight")
