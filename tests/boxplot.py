import scanpy as sc
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=100, facecolor="white")

# cellbin_moran/io/file_operations.py

import os
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import seaborn as sns
from scipy.stats import mannwhitneyu
from typing import Dict
from scipy.stats import zscore


# your_project/io/data_loading.py


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

merged_map = {
 'Secondary motor area': 'PFC',
 'Secondary motor area layer 1': 'PFC',
 'Secondary motor area layer 2/3': 'PFC',
 'Secondary motor area layer 5': 'PFC',
 'Secondary motor area layer 6a': 'PFC',
 'Secondary motor area layer 6b': 'PFC',
 'Anterior cingulate area': 'PFC',
 'Anterior cingulate area layer 1': 'PFC',
 'Anterior cingulate area layer 2/3': 'PFC',
 'Anterior cingulate area layer 5': 'PFC',
 'Anterior cingulate area layer 6a': 'PFC',
 'Anterior cingulate area layer 6b': 'PFC',
 'Anterior cingulate area dorsal part': 'PFC',
 'Anterior cingulate area dorsal part layer 1': 'PFC',
 'Anterior cingulate area dorsal part layer 2/3': 'PFC',
 'Anterior cingulate area dorsal part layer 5': 'PFC',
 'Anterior cingulate area dorsal part layer 6a': 'PFC',
 'Anterior cingulate area dorsal part layer 6b': 'PFC',
 'Anterior cingulate area ventral part': 'PFC',
 'Anterior cingulate area ventral part layer 1': 'PFC',
 'Anterior cingulate area ventral part layer 2/3': 'PFC',
 'Anterior cingulate area ventral part layer 5': 'PFC',
 'Anterior cingulate area ventral part 6a': 'PFC',
 'Anterior cingulate area ventral part 6b': 'PFC',
 'Prelimbic area': 'PFC',
 'Prelimbic area layer 1': 'PFC',
 'Prelimbic area layer 2': 'PFC',
 'Prelimbic area layer 2/3': 'PFC',
 'Prelimbic area layer 5': 'PFC',
 'Prelimbic area layer 6a': 'PFC',
 'Prelimbic area layer 6b': 'PFC',
 'Striatum dorsal region': 'STR',
 'Caudoputamen': 'STR',
 'Striatum ventral region': 'STR',
 'Nucleus accumbens': 'STR',
 'Islands of Calleja': 'STR',
 'Major island of Calleja': 'STR',
 'Olfactory tubercle layers 1-3': 'STR',
 'Olfactory tubercle molecular layer': 'STR',
 'Olfactory tubercle pyramidal layer': 'STR',
 'Olfactory tubercle polymorph layer': 'STR',
 'Lateral strip of striatum': 'STR',
 'Lateral septal complex': 'STR',
 'Lateral septal nucleus': 'STR',
 'Lateral septal nucleus caudal (caudodorsal) part': 'STR',
 'Lateral septal nucleus rostral (rostroventral) part': 'STR',
 'Lateral septal nucleus ventral part': 'STR',
 'Septofimbrial nucleus': 'STR',
 'Septohippocampal nucleus': 'STR',
 'Striatum-like amygdalar nuclei': 'STR',
 'Anterior amygdalar area': 'STR',
 'Bed nucleus of the accessory olfactory tract': 'STR',
 'Central amygdalar nucleus': 'STR',
 'Central amygdalar nucleus capsular part': 'STR',
 'Central amygdalar nucleus lateral part': 'STR',
 'Central amygdalar nucleus medial part': 'STR',
 'Intercalated amygdalar nucleus': 'STR',
 'Medial amygdalar nucleus': 'STR',
 'Medial amygdalar nucleus anterodorsal part': 'STR',
 'Medial amygdalar nucleus anteroventral part': 'STR',
 'Medial amygdalar nucleus posterodorsal part': 'STR',
 'Medial amygdalar nucleus posterodorsal part sublayer a': 'STR',
 'Medial amygdalar nucleus posterodorsal part sublayer b': 'STR',
 'Medial amygdalar nucleus posterodorsal part sublayer c': 'STR',
 'Medial amygdalar nucleus posteroventral part': 'STR',
 'Triangular nucleus of septum': 'STR',
 'fimbria': 'STR',
 'column of fornix': 'STR',
 'Substantia innominata': 'BF',
 'Magnocellular nucleus': 'BF',
 'Diagonal band nucleus': 'BF',
 'Hypothalamus': 'BF',
 'Periventricular zone': 'BF',
 'Supraoptic nucleus': 'BF',
 'Accessory supraoptic group': 'BF',
 'Nucleus circularis': 'BF',
 'Paraventricular hypothalamic nucleus': 'BF',
 'Paraventricular hypothalamic nucleus magnocellular division': 'BF',
 'Paraventricular hypothalamic nucleus magnocellular division anterior magnocellular part': 'BF',
 'Paraventricular hypothalamic nucleus magnocellular division medial magnocellular part': 'BF',
 'Paraventricular hypothalamic nucleus magnocellular division posterior magnocellular part': 'BF',
 'Paraventricular hypothalamic nucleus magnocellular division posterior magnocellular part lateral zone': 'BF',
 'Paraventricular hypothalamic nucleus magnocellular division posterior magnocellular part medial zone': 'BF',
 'Paraventricular hypothalamic nucleus parvicellular division': 'BF',
 'Paraventricular hypothalamic nucleus parvicellular division anterior parvicellular part': 'BF',
 'Paraventricular hypothalamic nucleus parvicellular division medial parvicellular part dorsal zone': 'BF',
 'Paraventricular hypothalamic nucleus parvicellular division periventricular part': 'BF',
 'Periventricular hypothalamic nucleus anterior part': 'BF',
 'Periventricular hypothalamic nucleus intermediate part': 'BF',
 'Arcuate hypothalamic nucleus': 'BF',
 'Periventricular region': 'BF',
 'Anterodorsal preoptic nucleus': 'BF',
 'Anterior hypothalamic area': 'BF',
 'Anteroventral preoptic nucleus': 'BF',
 'Anteroventral periventricular nucleus': 'BF',
 'Dorsomedial nucleus of the hypothalamus': 'BF',
 'Dorsomedial nucleus of the hypothalamus anterior part': 'BF',
 'Dorsomedial nucleus of the hypothalamus posterior part': 'BF',
 'Dorsomedial nucleus of the hypothalamus ventral part': 'BF',
 'Medial preoptic area': 'BF',
 'Vascular organ of the lamina terminalis': 'BF',
 'Posterodorsal preoptic nucleus': 'BF',
 'Parastrial nucleus': 'BF',
 'Suprachiasmatic preoptic nucleus': 'BF',
 'Periventricular hypothalamic nucleus posterior part': 'BF',
 'Periventricular hypothalamic nucleus preoptic part': 'BF',
 'Subparaventricular zone': 'BF',
 'Suprachiasmatic nucleus': 'BF',
 'Subfornical organ': 'BF',
 'Ventromedial preoptic nucleus': 'BF',
 'Ventrolateral preoptic nucleus': 'BF',
 'Hypothalamic medial zone': 'BF',
 'Anterior hypothalamic nucleus': 'BF',
 'Anterior hypothalamic nucleus anterior part': 'BF',
 'Anterior hypothalamic nucleus central part': 'BF',
 'Anterior hypothalamic nucleus dorsal part': 'BF',
 'Anterior hypothalamic nucleus posterior part': 'BF',
 'Mammillary body': 'BF',
 'Lateral mammillary nucleus': 'BF',
 'Medial mammillary nucleus': 'BF',
 'Medial mammillary nucleus median part': 'BF',
 'Medial mammillary nucleus lateral part': 'BF',
 'Medial mammillary nucleus medial part': 'BF',
 'Medial mammillary nucleus posterior part': 'BF',
 'Medial mammillary nucleus dorsal part': 'BF',
 'Supramammillary nucleus': 'BF',
 'Supramammillary nucleus lateral part': 'BF',
 'Supramammillary nucleus medial part': 'BF',
 'Tuberomammillary nucleus': 'BF',
 'Tuberomammillary nucleus dorsal part': 'BF',
 'Tuberomammillary nucleus ventral part': 'BF',
 'Medial preoptic nucleus': 'BF',
 'Medial preoptic nucleus central part': 'BF',
 'Medial preoptic nucleus lateral part': 'BF',
 'Medial preoptic nucleus medial part': 'BF',
 'Dorsal premammillary nucleus': 'BF',
 'Ventral premammillary nucleus': 'BF',
 'Paraventricular hypothalamic nucleus descending division': 'BF',
 'Paraventricular hypothalamic nucleus descending division dorsal parvicellular part': 'BF',
 'Paraventricular hypothalamic nucleus descending division forniceal part': 'BF',
 'Paraventricular hypothalamic nucleus descending division lateral parvicellular part': 'BF',
 'Paraventricular hypothalamic nucleus descending division medial parvicellular part ventral zone': 'BF',
 'Ventromedial hypothalamic nucleus': 'BF',
 'Ventromedial hypothalamic nucleus anterior part': 'BF',
 'Ventromedial hypothalamic nucleus central part': 'BF',
 'Ventromedial hypothalamic nucleus dorsomedial part': 'BF',
 'Ventromedial hypothalamic nucleus ventrolateral part': 'BF',
 'Posterior hypothalamic nucleus': 'BF',
 'Hypothalamic lateral zone': 'BF',
 'Lateral hypothalamic area': 'BF',
 'Lateral preoptic area': 'BF',
 'Preparasubthalamic nucleus': 'BF',
 'Parasubthalamic nucleus': 'BF',
 'Perifornical nucleus': 'BF',
 'Retrochiasmatic area': 'BF',
 'Subthalamic nucleus': 'BF',
 'Tuberal nucleus': 'BF',
 'Zona incerta': 'BF',
 'Dopaminergic A13 group': 'BF',
 'Fields of Forel': 'BF',
 'Median eminence': 'BF',
 'Retrosplenial area': 'RS',
 'Retrosplenial area lateral agranular part': 'RS',
 'Retrosplenial area lateral agranular part layer 1': 'RS',
 'Retrosplenial area lateral agranular part layer 2/3': 'RS',
 'Retrosplenial area lateral agranular part layer 5': 'RS',
 'Retrosplenial area lateral agranular part layer 6a': 'RS',
 'Retrosplenial area lateral agranular part layer 6b': 'RS',
 'Mediomedial anterior visual area': 'RS',
 'Mediomedial anterior visual area layer 1': 'RS',
 'Mediomedial anterior visual area layer 2/3': 'RS',
 'Mediomedial anterior visual area layer 4': 'RS',
 'Mediomedial anterior visual arealayer 5': 'RS',
 'Mediomedial anterior visual area layer 6a': 'RS',
 'Mediomedial anterior visual area layer 6b': 'RS',
 'Mediomedial posterior visual area': 'RS',
 'Mediomedial posterior visual area layer 1': 'RS',
 'Mediomedial posterior visual area layer 2/3': 'RS',
 'Mediomedial posterior visual area layer 4': 'RS',
 'Mediomedial posterior visual arealayer 5': 'RS',
 'Mediomedial posterior visual area layer 6a': 'RS',
 'Mediomedial posterior visual area layer 6b': 'RS',
 'Medial visual area': 'RS',
 'Medial visual area layer 1': 'RS',
 'Medial visual area layer 2/3': 'RS',
 'Medial visual area layer 4': 'RS',
 'Medial visual arealayer 5': 'RS',
 'Medial visual area layer 6a': 'RS',
 'Medial visual area layer 6b': 'RS',
 'Retrosplenial area dorsal part': 'RS',
 'Retrosplenial area dorsal part layer 1': 'RS',
 'Retrosplenial area dorsal part layer 2/3': 'RS',
 'Retrosplenial area dorsal part layer 4': 'RS',
 'Retrosplenial area dorsal part layer 5': 'RS',
 'Retrosplenial area dorsal part layer 6a': 'RS',
 'Retrosplenial area dorsal part layer 6b': 'RS',
 'Retrosplenial area ventral part': 'RS',
 'Retrosplenial area ventral part layer 1': 'RS',
 'Retrosplenial area ventral part layer 2': 'RS',
 'Retrosplenial area ventral part layer 2/3': 'RS',
 'Retrosplenial area ventral part layer 5': 'RS',
 'Retrosplenial area ventral part layer 6a': 'RS',
 'Retrosplenial area ventral part layer 6b': 'RS',
 'Ectorhinal area': 'ECT',
 'Ectorhinal area/Layer 1': 'ECT',
 'Ectorhinal area/Layer 2/3': 'ECT',
 'Ectorhinal area/Layer 5': 'ECT',
 'Ectorhinal area/Layer 6a': 'ECT',
 'Ectorhinal area/Layer 6b': 'ECT',
 'Perirhinal area': 'ECT',
 'Perirhinal area layer 1': 'ECT',
 'Perirhinal area layer 2/3': 'ECT',
 'Perirhinal area layer 5': 'ECT',
 'Perirhinal area layer 6a': 'ECT',
 'Perirhinal area layer 6b': 'ECT',
 'Entorhinal area': 'ECT',
 'Entorhinal area lateral part': 'ECT',
 'Entorhinal area lateral part layer 1': 'ECT',
 'Entorhinal area lateral part layer 2': 'ECT',
 'Entorhinal area lateral part layer 2/3': 'ECT',
 'Entorhinal area lateral part layer 2a': 'ECT',
 'Entorhinal area lateral part layer 2b': 'ECT',
 'Entorhinal area lateral part layer 3': 'ECT',
 'Entorhinal area lateral part layer 4': 'ECT',
 'Entorhinal area lateral part layer 4/5': 'ECT',
 'Entorhinal area lateral part layer 5': 'ECT',
 'Entorhinal area lateral part layer 5/6': 'ECT',
 'Entorhinal area lateral part layer 6a': 'ECT',
 'Entorhinal area lateral part layer 6b': 'ECT',
 'Entorhinal area medial part dorsal zone': 'ECT',
 'Entorhinal area medial part dorsal zone layer 1': 'ECT',
 'Entorhinal area medial part dorsal zone layer 2': 'ECT',
 'Entorhinal area medial part dorsal zone layer 2a': 'ECT',
 'Entorhinal area medial part dorsal zone layer 2b': 'ECT',
 'Entorhinal area medial part dorsal zone layer 3': 'ECT',
 'Entorhinal area medial part dorsal zone layer 4': 'ECT',
 'Entorhinal area medial part dorsal zone layer 5': 'ECT',
 'Entorhinal area medial part dorsal zone layer 5/6': 'ECT',
 'Entorhinal area medial part dorsal zone layer 6': 'ECT',
 'Entorhinal area medial part ventral zone': 'ECT',
 'Entorhinal area medial part ventral zone layer 1': 'ECT',
 'Entorhinal area medial part ventral zone layer 2': 'ECT',
 'Entorhinal area medial part ventral zone layer 3': 'ECT',
 'Entorhinal area medial part ventral zone layer 4': 'ECT',
 'Entorhinal area medial part ventral zone layer 5/6': 'ECT',
 'Hippocampal formation': 'HPF',
 'Hippocampal region': 'HPF',
 "Ammon's horn": 'HPF',
 'Field CA1': 'HPF',
 'Field CA1 stratum lacunosum-moleculare': 'HPF',
 'Field CA1 stratum oriens': 'HPF',
 'Field CA1 pyramidal layer': 'HPF',
 'Field CA1 stratum radiatum': 'HPF',
 'Field CA2': 'HPF',
 'Field CA2 stratum lacunosum-moleculare': 'HPF',
 'Field CA2 stratum oriens': 'HPF',
 'Field CA2 pyramidal layer': 'HPF',
 'Field CA2 stratum radiatum': 'HPF',
 'Field CA3': 'HPF',
 'Field CA3 stratum lacunosum-moleculare': 'HPF',
 'Field CA3 stratum lucidum': 'HPF',
 'Field CA3 stratum oriens': 'HPF',
 'Field CA3 pyramidal layer': 'HPF',
 'Field CA3 stratum radiatum': 'HPF',
 'Dentate gyrus': 'HPF',
 'Dentate gyrus molecular layer': 'HPF',
 'Dentate gyrus polymorph layer': 'HPF',
 'Dentate gyrus granule cell layer': 'HPF',
 'Dentate gyrus subgranular zone': 'HPF',
 'Dentate gyrus crest': 'HPF',
 'Dentate gyrus crest molecular layer': 'HPF',
 'Dentate gyrus crest polymorph layer': 'HPF',
 'Dentate gyrus crest granule cell layer': 'HPF',
 'Dentate gyrus lateral blade': 'HPF',
 'Dentate gyrus lateral blade molecular layer': 'HPF',
 'Dentate gyrus lateral blade polymorph layer': 'HPF',
 'Dentate gyrus lateral blade granule cell layer': 'HPF',
 'Dentate gyrus medial blade': 'HPF',
 'Dentate gyrus medial blade molecular layer': 'HPF',
 'Dentate gyrus medial blade polymorph layer': 'HPF',
 'Dentate gyrus medial blade granule cell layer': 'HPF',
 'Fasciola cinerea': 'HPF',
 'Induseum griseum': 'HPF',
 'Retrohippocampal region': 'HPF',
 'Parasubiculum': 'HPF',
 'Parasubiculum layer 1': 'HPF',
 'Parasubiculum layer 2': 'HPF',
 'Parasubiculum layer 3': 'HPF',
 'Postsubiculum': 'HPF',
 'Postsubiculum layer 1': 'HPF',
 'Postsubiculum layer 2': 'HPF',
 'Postsubiculum layer 3': 'HPF',
 'Presubiculum': 'HPF',
 'Presubiculum layer 1': 'HPF',
 'Presubiculum layer 2': 'HPF',
 'Presubiculum layer 3': 'HPF',
 'Subiculum': 'HPF',
 'Subiculum dorsal part': 'HPF',
 'Subiculum dorsal part molecular layer': 'HPF',
 'Subiculum dorsal part pyramidal layer': 'HPF',
 'Subiculum dorsal part stratum radiatum': 'HPF',
 'Subiculum ventral part': 'HPF',
 'Subiculum ventral part molecular layer': 'HPF',
 'Subiculum ventral part pyramidal layer': 'HPF',
 'Subiculum ventral part stratum radiatum': 'HPF',
 'Prosubiculum': 'HPF',
 'Prosubiculum dorsal part': 'HPF',
 'Prosubiculum dorsal part molecular layer': 'HPF',
 'Prosubiculum dorsal part pyramidal layer': 'HPF',
 'Prosubiculum dorsal part stratum radiatum': 'HPF',
 'Prosubiculum ventral part': 'HPF',
 'Prosubiculum ventral part molecular layer': 'HPF',
 'Prosubiculum ventral part pyramidal layer': 'HPF',
 'Prosubiculum ventral part stratum radiatum': 'HPF',
 'Hippocampo-amygdalar transition area': 'HPF',
 'Area prostriata': 'HPF',
 'Substantia innominata': 'BF',
 'Magnocellular nucleus': 'BF',
 'Diagonal band nucleus': 'BF',
 'Thalamus': 'TH',
 'Thalamus sensory-motor cortex related': 'TH',
 'Ventral group of the dorsal thalamus': 'TH',
 'Ventral anterior-lateral complex of the thalamus': 'TH',
 'Ventral medial nucleus of the thalamus': 'TH',
 'Ventral posterior complex of the thalamus': 'TH',
 'Ventral posterolateral nucleus of the thalamus': 'TH',
 'Ventral posterolateral nucleus of the thalamus parvicellular part': 'TH',
 'Ventral posteromedial nucleus of the thalamus': 'TH',
 'Ventral posteromedial nucleus of the thalamus parvicellular part': 'TH',
 'Posterior triangular thalamic nucleus': 'TH',
 'Subparafascicular nucleus': 'TH',
 'Subparafascicular nucleus magnocellular part': 'TH',
 'Subparafascicular nucleus parvicellular part': 'TH',
 'Subparafascicular area': 'TH',
 'Peripeduncular nucleus': 'TH',
 'Geniculate group dorsal thalamus': 'TH',
 'Medial geniculate complex': 'TH',
 'Medial geniculate complex dorsal part': 'TH',
 'Medial geniculate complex ventral part': 'TH',
 'Medial geniculate complex medial part': 'TH',
 'Dorsal part of the lateral geniculate complex': 'TH',
 'Dorsal part of the lateral geniculate complex shell': 'TH',
 'Dorsal part of the lateral geniculate complex core': 'TH',
 'Dorsal part of the lateral geniculate complex ipsilateral zone': 'TH',
 'Thalamus polymodal association cortex related': 'TH',
 'Lateral group of the dorsal thalamus': 'TH',
 'Lateral posterior nucleus of the thalamus': 'TH',
 'Posterior complex of the thalamus': 'TH',
 'Posterior limiting nucleus of the thalamus': 'TH',
 'Suprageniculate nucleus': 'TH',
 'Ethmoid nucleus of the thalamus': 'TH',
 'Retroethmoid nucleus': 'TH',
 'Anterior group of the dorsal thalamus': 'TH',
 'Anteroventral nucleus of thalamus': 'TH',
 'Anteromedial nucleus': 'TH',
 'Anteromedial nucleus dorsal part': 'TH',
 'Anteromedial nucleus ventral part': 'TH',
 'Anterodorsal nucleus': 'TH',
 'Interanteromedial nucleus of the thalamus': 'TH',
 'Interanterodorsal nucleus of the thalamus': 'TH',
 'Lateral dorsal nucleus of thalamus': 'TH',
 'Medial group of the dorsal thalamus': 'TH',
 'Intermediodorsal nucleus of the thalamus': 'TH',
 'Mediodorsal nucleus of thalamus': 'TH',
 'Mediodorsal nucleus of the thalamus central part': 'TH',
 'Mediodorsal nucleus of the thalamus lateral part': 'TH',
 'Mediodorsal nucleus of the thalamus medial part': 'TH',
 'Submedial nucleus of the thalamus': 'TH',
 'Perireunensis nucleus': 'TH',
 'Midline group of the dorsal thalamus': 'TH',
 'Paraventricular nucleus of the thalamus': 'TH',
 'Parataenial nucleus': 'TH',
 'Nucleus of reuniens': 'TH',
 'Xiphoid thalamic nucleus': 'TH',
 'Intralaminar nuclei of the dorsal thalamus': 'TH',
 'Rhomboid nucleus': 'TH',
 'Central medial nucleus of the thalamus': 'TH',
 'Paracentral nucleus': 'TH',
 'Central lateral nucleus of the thalamus': 'TH',
 'Parafascicular nucleus': 'TH',
 'Posterior intralaminar thalamic nucleus': 'TH',
 'Reticular nucleus of the thalamus': 'TH',
 'Geniculate group ventral thalamus': 'TH',
 'Intergeniculate leaflet of the lateral geniculate complex': 'TH',
 'Intermediate geniculate nucleus': 'TH',
 'Ventral part of the lateral geniculate complex': 'TH',
 'Ventral part of the lateral geniculate complex lateral zone': 'TH',
 'Ventral part of the lateral geniculate complex medial zone': 'TH',
 'Subgeniculate nucleus': 'TH',
 'Epithalamus': 'TH',
 'Medial habenula': 'TH',
 'Lateral habenula': 'TH',
 'Pineal body': 'TH',
            }

spot_dir = "/home1/jijh/st_project/bin50_analysis/seurat_related"
spot_dic = list_files_matching_criteria(spot_dir , condition=f"'h5ad' in file")

adatas = load_data_in_parallel(spot_dic, load_sct_and_set_index)
cellbin_meta = read_and_process_metadata("/home1/jijh/st_project/cellbin_analysis/annotated_cell_bins/cellbin_meta/", "'csv' in file")



gene_dfs = {}
errors = {}

for key, adata in adatas.items():
    print(f"Processing {key}")
    try:
        adata_cp = adata.raw.to_adata()
    
        sc.pp.normalize_total(adata_cp, target_sum=1e4)
        sc.pp.log1p(adata_cp)
        for gene in genes:
            value = np.array(adata_cp[:,gene].X.todense())
            adata_cp.obs[gene] = value
    
        df = adata_cp.obs.copy()
        tar_df = adj_cb_meta[key].copy()
        tar_tree = cKDTree(tar_df[['x', 'y']])
        que_points = np.array(df[['x', 'y']])
        dist, idx = tar_tree.query(que_points)
        df['fine'] = tar_df["fine"].iloc[idx].values
    
    
        for gene in genes:
            mask = df[gene] > df[gene].min()
            df[f"{gene}_fil"] = df[gene]
            df[f"{gene}_fil"][~mask] = None
            df[f"{gene}_z"] = zscore(df[f"{gene}_fil"], nan_policy = "omit")
            data = df[f"{gene}_fil"]
            scale_data = (data - data.min()) / (data.max() - data.min())
            df[f"{gene}_scale"] = scale_data
    
        gene_dfs[key] = df
    except Exception as exc:
        print("Error for " + key)
        errors[key] = exc

merged_df = pd.concat(gene_dfs.values())
micro_groups = {
    "PFC": "group2",
    "ENT": "group2",
    "ECT": "group2",
    "HPF": "group2",
    "TH": "group2",
    "RS": "group2",
    "BF": "group1",
    "STR": "group1",
}



merged_df["region_abb"] = merged_df["safe_name"].map(merged_map)
merged_df["group"] = merged_df["region_abb"].map(micro_groups)

unique_df = merged_df.copy()
unique_df = unique_df.reset_index()

# 固定 x 轴长度
fixed_xlim = [-2, 5]  # 可以根据具体数据调整

for gene in genes:
    mask = unique_df[gene] > unique_df[gene].min()
    
    # Calculate Mann-Whitney U test
    mask1 = unique_df['group'] == 'group1'
    mask1 &= unique_df[f"{gene}_z"].notna()
    group1 = unique_df[mask1][f"{gene}_z"]
    
    mask2 = unique_df['group'] == 'group2'
    mask2 &= unique_df[f"{gene}_z"].notna()
    group2 = unique_df[mask2][f"{gene}_z"]
    stat, p_value = mannwhitneyu(group1, group2)
    
    # Determine significance level and assign asterisk
    if p_value < 0.001:
        sig_level = '***'
    elif p_value < 0.01:
        sig_level = '**'
    elif p_value < 0.05:
        sig_level = '*'
    else:
        sig_level = 'ns'
    
    # Calculate mean values for each group
    mean_group1 = group1.mean()
    mean_group2 = group2.mean()
    
    # Create horizontal boxplot
    plt.figure(figsize=(5, 2))
    ax = sns.boxplot(data=unique_df[mask], y='group', x=f"{gene}_z", hue="group", boxprops=dict(alpha=.5), 
                     dodge=False, showfliers=False, legend=None)
    ax.vlines(x=0, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], linestyle="dashed", color="red")
    
    # Annotate mean values on the plot
    for i, mean in enumerate([mean_group1, mean_group2]):
        ax.text(fixed_xlim[1], i, f'{mean:.2f}', ha='left', va='center', color='black', weight='bold')
    
    # Customize the plot
    plt.title(f"Distribution of {gene} expression\nMann-Whitney U test p-value = {p_value:.4f}\n({sig_level})", fontsize=8, weight='bold')
    plt.xlabel("Expression", fontsize=12)
    plt.ylabel('Group', fontsize=12)
    plt.grid(False)
    
    # Set fixed x-axis limits
    plt.xlim(fixed_xlim)
    
    # Adjust tight layout for the legend
    plt.tight_layout(rect=[0, 0, 0.75, 1])

    plt.savefig(f"./expression_slice/all_sample{gene}.pdf", dpi=350)
    
    # Show the plot
    plt.show()
