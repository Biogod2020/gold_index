# cellbin_moran/io/file_operations.py

import os
import re
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from typing import Dict

def list_files_matching_criteria(directory: str, condition: str = None, regex: str = None, separator: str = "_") -> dict:
    """
    Lists files in a directory matching a given condition or regular expression.

    Args:
        directory: The directory to search for files.
        condition: The conditional equation to filter files by. The condition should be a valid Python expression
                   where 'file' can be used as the variable.
        regex: The regular expression to filter files by.
        separator: The separator used to split the filenames.

    Returns:
        A dictionary where the keys are the prefixes of the filenames split by the separator
        and the values are the full file paths of the files that match the condition or regex.
    """
    files = sorted(os.listdir(directory))
    paths = {file.split(separator)[0]: os.path.join(directory, file) for file in files}
    
    if condition:
        filtered_paths = {file: path for file, path in paths.items() if eval(condition)}
    elif regex:
        pattern = re.compile(regex)
        filtered_paths = {file: path for file, path in paths.items() if pattern.search(file)}
    else:
        filtered_paths = paths
    
    return filtered_paths


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
