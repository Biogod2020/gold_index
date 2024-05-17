#!/bin/zsh

# Create the main project directories
mkdir -p cellbin_moran/analysis
mkdir -p cellbin_moran/io
mkdir -p tests

# Create the __init__.py files to make the directories Python packages
touch cellbin_moran/__init__.py
touch cellbin_moran/analysis/__init__.py
touch cellbin_moran/io/__init__.py
touch tests/__init__.py

# Create README.md with basic content
cat <<EOT > README.md
# Your Project

This project is designed for single-cell data analysis.

## Structure

- \`cellbin_moran/\`: Main package directory.
- \`cellbin_moran/analysis/\`: Contains functions for spatial analysis.
- \`cellbin_moran/io/\`: Contains functions for data loading and file operations.
- \`tests/\`: Contains unit tests for the functions.
EOT

# Create setup.py with basic content
cat <<EOT > setup.py
from setuptools import setup, find_packages

setup(
    name='cellbin_moran',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scanpy',
        'squidpy',
        # other dependencies
    ],
)
EOT

# Create requirements.txt with necessary dependencies
cat <<EOT > requirements.txt
numpy
pandas
scanpy
squidpy
EOT

# Create cellbin_moran/analysis/spatial_analysis.py with the provided functions
cat <<EOT > cellbin_moran/analysis/spatial_analysis.py
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
    key (str): Optional. Key under which the batch information is stored in `obs`.
    
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
EOT

# Create cellbin_moran/io/data_loading.py with the provided function
cat <<EOT > cellbin_moran/io/data_loading.py
import scanpy as sc

def load_sct_and_set_index(adata_path):
    adata = sc.read_h5ad(adata_path)
    adata.raw.var.set_index("_index", inplace=True)
    adata.var.set_index("_index", inplace=True)
    return adata
EOT

# Create cellbin_moran/io/file_operations.py with the provided functions
cat <<EOT > cellbin_moran/io/file_operations.py
import os
from concurrent.futures import ProcessPoolExecutor

def list_files_matching_criteria(directory, criteria):
    files = sorted(os.listdir(directory))
    paths = {file.split(sep="_")[0]: os.path.join(directory, file) for file in files}
    filtered_paths = {file: path for file, path in paths.items() if criteria in file}
    return filtered_paths

def load_data_in_parallel(file_paths, load_function):
    data = {}
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(load_function, path): file for file, path in file_paths.items()}
        for future in futures:
            try:
                data[futures[future]] = future.result()
            except Exception as exc:
                print(f"Error loading file {futures[future]}: {exc}")
    return data
EOT

# Create empty test files
touch tests/test_spatial_analysis.py
touch tests/test_data_loading.py
touch tests/test_file_operations.py

# Initialize Git (if not already initialized)
if [ ! -d .git ]; then
    git init
fi

# Add and commit the initial project structure to Git
git add .
git commit -m "Initialize project structure with required files and directories"
