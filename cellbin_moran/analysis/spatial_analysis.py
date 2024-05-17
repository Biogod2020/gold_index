import pandas as pd
import squidpy as sq
import numpy as np
import scanpy as sc
import anndata as ad

def compute_moranI(adata, genes, category="celltype", spatial_key="X_umap"):
    top_level_types = adata.obs[category].unique()
    result_df = pd.DataFrame()
    data_present = False  # Flag to track if any cell type passed the threshold
    
    for cell_type in top_level_types:
        mask = adata.obs[category] == cell_type
        num_cell = sum(mask)
        if num_cell > 10:
            print(f"{cell_type}: {num_cell}")
            sub_adata = adata[mask].copy()
            sq.gr.spatial_neighbors(sub_adata, n_neighs=6, spatial_key=spatial_key)
            sq.gr.spatial_autocorr(sub_adata, mode='moran', genes=genes, n_perms=100, n_jobs=1, attr="obs")
            moranI_df = sub_adata.uns['moranI']
            moranI_df[category] = cell_type
            moranI_df["num_cell"] = num_cell
            result_df = pd.concat([result_df, moranI_df])
            data_present = True  # Set flag to True as we have added data to the result dataframe
    
    # After going through all types, check if we have added any data to result_df
    if not data_present:
        # Since no cell type met the threshold, return an empty DataFrame
        return pd.DataFrame()
    
    result_df = result_df.set_index(category)
    result_df = result_df.sort_values("I", ascending=False)
    return result_df

def concatenate_and_intersect(adata_list, key=None):
    """
    Concatenate a list of AnnData objects, keeping only the intersection of their variables.
    
    Parameters:
    adata_list (list of AnnData): List of AnnData objects to be concatenated.
    key (str): Optional. Key under which the batch information is stored in .
    
    Returns:
    AnnData: Concatenated AnnData object with intersected variables.
    """
    if not adata_list:
        raise ValueError("The list of AnnData objects is empty.")
    
    # Identify common variables across all AnnData objects
    common_vars = adata_list[0].var_names
    for adata in adata_list[1:]:
        common_vars = np.intersect1d(common_vars, adata.var_names)
    
    # Filter each AnnData object to keep only the common variables
    filtered_adatas = []
    for adata in adata_list:
        mask = [var_name in common_vars for var_name in adata.var_names]
        filtered_adatas.append(adata[:, mask])
    
    # Concatenate the filtered AnnData objects
    concatenated_adata = ad.concat(filtered_adatas, axis=0, join='outer', merge='same', label=key)
    
    return concatenated_adata
