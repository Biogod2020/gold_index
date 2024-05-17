# your_project/io/file_operations.py

import os
from concurrent.futures import ProcessPoolExecutor
def list_files_matching_criteria(directory: str, criteria: str, separator: str = "_") -> dict:
    """
    Lists files in a directory matching a given criteria.

    Args:
        directory: The directory to search for files.
        criteria: The criteria to filter files by.
        separator: The separator used to split the filenames.

    Returns:
        A dictionary where the keys are the prefixes of the filenames split by the separator
        and the values are the full file paths of the files that match the criteria.
    """
    files = sorted(os.listdir(directory))
    paths = {file.split(separator)[0]: os.path.join(directory, file) for file in files}
    filtered_paths = {file: path for file, path in paths.items() if criteria in file}
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
