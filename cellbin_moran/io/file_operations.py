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
