# cellbin_moran/io/file_operations.py

import os
import re
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from typing import Dict
import scanpy as sc
import os
import re

def list_files_matching_criteria(directory: str, condition: str = None, regex: str = None, separator: str = "_", id_index: int = 0) -> dict:
    """
    Lists files in a directory matching a given condition or regular expression and extracts an ID from the filename.

    Args:
        directory: The directory to search for files.
        condition: The conditional equation to filter files by. The condition should be a valid Python expression
                   where 'file' can be used as the variable.
        regex: The regular expression to filter files by.
        separator: The separator used to split the filenames.
        id_index: The index of the split result to be used as the dictionary key.

    Returns:
        A dictionary where the keys are the specified parts of the filenames (split by the separator)
        and the values are the full file paths of the files that match the condition or regex.
    """
    files = sorted(os.listdir(directory))

    # Filter files based on condition or regex
    if condition:
        filtered_files = [file for file in files if eval(condition)]
    elif regex:
        pattern = re.compile(regex)
        filtered_files = [file for file in files if pattern.search(file)]
    else:
        filtered_files = files


    # Create the dictionary with the specified part of the filenames as keys
    paths = {file.split(separator)[id_index]: os.path.join(directory, file) for file in filtered_files}
    
    return paths




def load_data_in_parallel(file_paths: dict, load_function: callable) -> dict:
    """
    Loads data from multiple files in parallel using a specified load function.

    Args:
        file_paths: A dictionary where the keys are identifiers and the values are file paths.
        load_function: The function to use for loading data from each file path.

    Returns:
        A dictionary where the keys are identifiers and the values are the loaded data.
    """
    data = {}
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(load_function, path): file for file, path in file_paths.items()}
        for future in futures:
            try:
                data[futures[future]] = future.result()
            except Exception as exc:
                print(f"Error loading file {futures[future]}: {exc}")
    return data


def read_and_process_metadata(directory: str, criteria: str) -> Dict[str, pd.DataFrame]:
    """
    Reads and processes metadata files in a directory matching a given criteria.

    Args:
        directory: The directory to search for metadata files.
        criteria: The criteria to filter metadata files by.

    Returns:
        A dictionary where the keys are identifiers and the values are processed metadata DataFrames.
    """
    meta_paths = list_files_matching_criteria(directory, criteria)
    meta_data = {}
    for key, value in meta_paths.items():
        try:
            df = pd.read_csv(value)
            df["celltype"] = df["fine"].str.split("-").str[0]
            df = df.applymap(lambda x: 'NA' if str(x).lower() == 'nan' else x)
            meta_data[key] = df
        except Exception as e:
            print(f"Error reading or processing metadata file '{value}': {e}")
    return meta_data

# cellbin_moranI/analysis/spatial_analysis.py

from typing import Dict
import anndata as ad

def merge_metadata(cellbin_data: Dict[str, ad.AnnData], meta_data: Dict[str, pd.DataFrame]) -> None:
    """
    Merges cellbin data with metadata, updating the `.obs` attribute of AnnData objects.

    Args:
        cellbin_data: A dictionary where keys are identifiers and values are AnnData objects.
        meta_data: A dictionary where keys are identifiers and values are metadata DataFrames.
    """
    for key in cellbin_data:
        if key in meta_data:
            try:
                cellbin_data[key].obs = meta_data[key]
            except Exception as e:
                print(f"Error merging data for key '{key}': {e}")
        else:
            print(f"Metadata for key '{key}' not found in the provided meta_data dictionary.")




def load_sct_and_set_index(adata_path: str) -> sc.AnnData:
    """
    Loads an AnnData object from a file and sets the index for the `raw.var` and `var` dataframes.

    Args:
        adata_path: Path to the .h5ad file containing the AnnData object.

    Returns:
        The AnnData object with updated indices for `raw.var` and `var`.
    """
    adata = sc.read_h5ad(adata_path)
    adata.raw.var.set_index("_index", inplace=True)
    adata.var.set_index("_index", inplace=True)
    return adata

def read_csv_files_concurrently(file_dict):
    """
    Reads multiple CSV files concurrently and returns a dictionary of DataFrames.

    :param file_dict: Dictionary where the key is a variable name and the value is the path to the CSV file.
    :return: Dictionary where the key is the variable name and the value is the corresponding DataFrame.
    """
    def load_csv(key, path):
        print(f"Reading {key}")
        return key, pd.read_csv(path, skiprows=[1])

    result_dict = {}
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_key = {executor.submit(load_csv, key, path): key for key, path in file_dict.items()}
        
        for future in concurrent.futures.as_completed(future_to_key):
            key, df = future.result()
            result_dict[key] = df
            
    return result_dict