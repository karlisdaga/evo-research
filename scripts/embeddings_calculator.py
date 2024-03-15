import pandas as pd
import numpy as np
import os
import sys
import argparse

sys.path.append("/../src/")

from ablang_model import Ablang
from antiberty_model import Antiberty
from ESM_model import ESM
from sapiens_model import Sapiens
from protbert import ProtBert
from ESM_prob_model import ESM_prob
parser = argparse.ArgumentParser( )

parser.add_argument('-d','--dataset',required=False)   
parser.add_argument('--mode') 
parser.add_argument('--file_path', required=False)

args = parser.parse_args()

dataset = args.dataset
mode = args.mode

init_list = [ESM, ProtBert, Ablang]
suffixes = ["ESM", "protbert", "ablang"]

if mode == "general":
 
    repertoire_file  = pd.read_csv(args.file_path, delimiter=";")

    repertoire_file_folder = os.path.dirname(args.file_path)
    save_path = os.path.join(repertoire_file_folder,"embeddings")

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    starts = [0]*repertoire_file.shape[0]
    ends = repertoire_file["full_sequence"].apply(len)

    for i,model in enumerate(init_list): 

        if suffixes[i] == "ablang":
            model = model(chain = "heavy")
            per_pos = model.calc_evo_likelihood_matrix_per_position(sequences=list(repertoire_file["full_sequence"]))
        elif suffixes[i] == "sapiens":
            model = model(chain_type = "H")
            per_pos = model.calc_probability_matrix(sequence=str(repertoire_file["full_sequence"]))
        else:
            model = model()

        embeds = model.fit_transform(sequences=list(repertoire_file["full_sequence"]),starts=list(starts),ends=list(ends))
        embeds = pd.concat([repertoire_file,embeds],axis=1)
#Call model embeddings, per residue evo_likelihoods and best sequences
        embeds.to_csv(os.path.join(save_path,f"embeddings_{suffixes[i]}.csv"), index=False)   
        per_pos = model.calc_evo_likelihood_matrix_per_position(sequences=list(repertoire_file["full_sequence"]))
        best_evo = model.process_sequences(sequences=list(repertoire_file["full_sequence"]),starts=list(starts),ends=list(ends))
        best_evo.to_csv(os.path.join(save_path,f"best_sequence_evo.csv"), index = False)
else:
    data_folder_path = os.path.join("..","..","..","data",dataset,"VDJ")

    columns_to_save = ["barcode","contig_id","chain","v_gene","d_gene","j_gene","c_gene","raw_clonotype_id","raw_consensus_id"]

    for sample in os.listdir(data_folder_path):
        
        cellranger_path = os.path.join(data_folder_path, sample)
        # cellranger_path = os.path.join(cellranger_path, os.listdir(cellranger_path)[0])

        if not (os.path.isdir(os.path.join(cellranger_path,"embeddings"))):
            os.mkdir(os.path.join(cellranger_path,"embeddings"))

        embeddings_folder = os.path.join(cellranger_path,"embeddings")
        repertoire_file_path = os.path.join(cellranger_path,"filtered_contig_annotations.csv")

        repertoire_file = pd.read_csv(repertoire_file_path)
        # repertoire_file = repertoire_file.iloc[:500,:]

        if (mode == "cdr3_only"):
            repertoire_file["full_sequence"] = repertoire_file["cdr3"]

            repertoire_file = repertoire_file.dropna(subset=["full_sequence"])

            if not (os.path.isdir(os.path.join(embeddings_folder,"cdr3_only"))):
                os.mkdir(os.path.join(embeddings_folder,"cdr3_only"))

            save_path = os.path.join(embeddings_folder,"cdr3_only")
        elif (mode == "full_VDJ" or mode == "cdr3_from_VDJ"):
            repertoire_file["full_sequence"] = repertoire_file["fwr1"] + repertoire_file["cdr1"] + repertoire_file["fwr2"] + \
                                            repertoire_file["cdr2"] + repertoire_file["fwr3"] + repertoire_file["cdr3"] + repertoire_file["fwr4"]
            
            repertoire_file = repertoire_file.dropna(subset=["full_sequence"])
            
            if not (os.path.isdir(os.path.join(embeddings_folder,"full_VDJ"))):
                os.mkdir(os.path.join(embeddings_folder,"full_VDJ"))

            save_path = os.path.join(embeddings_folder,"full_VDJ")
        
        if mode == "cdr3_from_VDJ":
            x = repertoire_file["fwr1"] + repertoire_file["cdr1"] + repertoire_file["fwr2"] + \
                                            repertoire_file["cdr2"] + repertoire_file["fwr3"]
            starts = x.apply(len)

            y = repertoire_file["fwr1"] + repertoire_file["cdr1"] + repertoire_file["fwr2"] + \
                                            repertoire_file["cdr2"] + repertoire_file["fwr3"] + repertoire_file["cdr3"]   
            ends = y.apply(len)

            if not (os.path.isdir(os.path.join(embeddings_folder,"cdr3_from_VDJ"))):
                os.mkdir(os.path.join(embeddings_folder,"cdr3_from_VDJ"))
            
            save_path = os.path.join(embeddings_folder,"cdr3_from_VDJ")

        else:
            starts = pd.Series([0]*repertoire_file.shape[0])
            ends = repertoire_file["full_sequence"].apply(len)
        
        save_path = os.path.abspath(save_path)

        
        for i,model in enumerate(init_list):

            save_filepath = os.path.join(save_path,f"embeddings_{suffixes[i]}.csv.gzip")

            if os.path.exists(save_filepath):
                continue
            
            if suffixes[i] in ["ablang","sapiens"]:
                is_heavy_chain = list(repertoire_file["chain"] == "IGH")
                is_light_chain = list(repertoire_file["chain"] != "IGH")
                if suffixes[i] == "ablang":
                    embeds_1 = Ablang(chain="heavy").fit_transform(sequences=list(repertoire_file[is_heavy_chain]["full_sequence"]),
                                                        starts=list(starts[is_heavy_chain]),ends=list(ends[is_heavy_chain]),
                                                        path = save_filepath)
                    
                    embeds_1 = pd.concat([repertoire_file.loc[is_heavy_chain,columns_to_save].reset_index(drop=True),embeds_1],axis=1)
                    
                    embeds_2 = Ablang(chain="light").fit_transform(sequences=list(repertoire_file[is_light_chain]["full_sequence"]),
                                                        starts=list(starts[is_light_chain]),ends=list(ends[is_light_chain]),
                                                        path = save_filepath)
                    
                    embeds_2 = pd.concat([repertoire_file.loc[is_light_chain,columns_to_save].reset_index(drop=True),embeds_2],axis=1)

                    embeds = pd.concat([embeds_1,embeds_2],axis=0)
                    
                if suffixes[i] == "sapiens":
                    embeds_1 = Sapiens(chain_type="H").fit_transform(sequences=(repertoire_file[is_heavy_chain]["full_sequence"]),
                                                        starts=list(starts[is_heavy_chain]),ends=list(ends[is_heavy_chain]),
                                                        path = save_filepath)

                    embeds_1 = pd.concat([repertoire_file.loc[is_heavy_chain,columns_to_save].reset_index(drop=True),embeds_1],axis=1)

                    embeds_2 = Sapiens(chain_type="L").fit_transform(sequences=(repertoire_file[is_light_chain]["full_sequence"]),
                                                        starts=list(starts[is_light_chain]),ends=list(ends[is_light_chain]),
                                                        path = save_filepath)
                    
                    embeds_2 = pd.concat([repertoire_file.loc[is_light_chain,columns_to_save].reset_index(drop=True),embeds_2],axis=1)

                    embeds = pd.concat([embeds_1,embeds_2],axis=0)

            else:
                model = model()
                embeds = model.fit_transform(sequences=list(repertoire_file["full_sequence"]),starts=list(starts),ends=list(ends),path = save_filepath)
                embeds = pd.concat([repertoire_file[columns_to_save],embeds],axis=1)

            embeds.to_csv(save_filepath, index=False, compression="gzip")
