# your_project/analysis/spatial_analysis.py

import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from sklearn.utils import check_random_state
from scipy.sparse import csr_matrix
from libpysal.weights import WSP
from esda.moran import Moran
from typing import List, Dict
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from scipy.spatial.distance import pdist, squareform
from anndata import AnnData
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import concurrent.futures
import warnings
import logging


# def compute_moranI(
#     adata: ad.AnnData, 
#     genes: list[str], 
#     category: str = "celltype", 
#     spatial_key: str = "X_umap"
# ) -> pd.DataFrame:
#     """
#     Computes Moran's I spatial autocorrelation for specified genes.

#     Args:
#         adata: The AnnData object to analyze.
#         genes: List of gene names to compute Moran's I for.
#         category: The categorical variable in `adata.obs` to group by.
#         spatial_key: The key in `adata.obsm` containing spatial coordinates.

#     Returns:
#         A DataFrame with Moran's I results for each cell type and gene.
#     """
#     top_level_types = adata.obs[category].unique()
#     result_df = pd.DataFrame()
#     data_present = False
    
#     for cell_type in top_level_types:
#         mask = adata.obs[category] == cell_type
#         num_cell = sum(mask)
#         if num_cell > 10:
#             print(f"{cell_type}: {num_cell}")
#             sub_adata = adata[mask].copy()
#             sq.gr.spatial_neighbors(sub_adata, n_neighs=6, spatial_key=spatial_key)
#             sq.gr.spatial_autocorr(sub_adata, mode='moran', genes=genes, n_perms=100, n_jobs=1, attr="obs")
#             moranI_df = sub_adata.uns['moranI']
#             moranI_df[category] = cell_type
#             moranI_df["num_cell"] = num_cell
#             result_df = pd.concat([result_df, moranI_df])
#             data_present = True
    
#     if not data_present:
#         return pd.DataFrame()
    
#     result_df = result_df.set_index(category)
#     result_df = result_df.sort_values("I", ascending=False)
#     return result_df

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def concatenate_and_intersect(
    adata_list: list[ad.AnnData], 
    key: str = None
) -> ad.AnnData:
    """
    Concatenates a list of AnnData objects, keeping only the intersection of their variables.

    Args:
        adata_list: List of AnnData objects to concatenate.
        key: Optional key under which the batch information is stored in `obs`.

    Returns:
        Concatenated AnnData object with intersected variables.
    """
    if not adata_list:
        raise ValueError("The list of AnnData objects is empty.")
    
    common_vars = adata_list[0].var_names
    for adata in adata_list[1:]:
        common_vars = np.intersect1d(common_vars, adata.var_names)
    
    filtered_adatas = []
    for adata in adata_list:
        mask = [var_name in common_vars for var_name in adata.var_names]
        filtered_adatas.append(adata[:, mask])
    
    concatenated_adata = ad.concat(filtered_adatas, axis=0, join='outer', merge='same', label=key)
    return concatenated_adata



def hierarchical_sample(
    adata: ad.AnnData,
    groupby_cols: list[str],
    n_samples: int | float | None = None,
    fraction: float | None = None,
    random_state: int | np.random.RandomState | None = None,
) -> ad.AnnData:
    """
    Performs hierarchical random sampling of an AnnData object based on multiple categorical variables.

    Args:
        adata: The AnnData object to sample from.
        groupby_cols: A list of column names from `adata.obs` to use for hierarchical grouping.
        n_samples: The number of samples to select from each group at the final level.
                   If a float, it's interpreted as the fraction of samples to keep.
        fraction: The fraction of samples to keep from each group at the final level.
                  If provided, `n_samples` is ignored.
        random_state: Random seed for reproducibility.

    Returns:
        A new AnnData object containing the sampled cells.
    """
    import anndata as ad
    import pandas as pd
    import numpy as np
    from sklearn.utils import check_random_state

    if (n_samples is not None and fraction is not None) or (
        n_samples is None and fraction is None
    ):
        raise ValueError("Must specify exactly one of 'n_samples' or 'fraction'.")

    rng = check_random_state(random_state)

    def _sample_group(group: pd.DataFrame, level: int) -> pd.DataFrame:
        """Recursively samples from each group."""
        if level == len(groupby_cols) - 1:
            # Sample at the final level
            if fraction is not None:
                return group.sample(frac=fraction, random_state=rng)
            elif isinstance(n_samples, int):
                return group.groupby(groupby_cols[level], group_keys=False, observed=False).apply(
                    lambda x: x.sample(n=min(n_samples, len(x)), random_state=rng)
                )
            elif isinstance(n_samples, float):
                return group.sample(frac=n_samples, random_state=rng)
        else:
            # Group by the next level and sample recursively
            return group.groupby(groupby_cols[level + 1], group_keys=False, observed=False).apply(
                lambda x: _sample_group(x, level + 1)
            )

    # Starting the hierarchical sampling from the top level
    sampled_obs = _sample_group(adata.obs, level=0)
    sampled_obs_index = sampled_obs.index.get_level_values(-1) if isinstance(sampled_obs.index, pd.MultiIndex) else sampled_obs.index

    return adata[sampled_obs_index, :].copy()


def subset_anndata(
    adata: 'anndata.AnnData', 
    conditions: dict
) -> 'anndata.AnnData':
    """
    Subsets the AnnData object to only include cells where the `.obs` column values match specified conditions.

    Args:
        adata: The AnnData object to subset.
        conditions: A dictionary where keys are column names from `adata.obs` and values are the values to match in those columns.

    Returns:
        A new AnnData object containing only the cells that match all specified conditions.
    """
    import anndata as ad
    import pandas as pd
    
    # Start with a boolean mask that includes all cells
    mask = pd.Series([True] * adata.shape[0], index=adata.obs.index)
    
    # Apply each condition to the mask
    for column, value in conditions.items():
        mask &= adata.obs[column] == value
    
    # Subset the AnnData object using the mask
    return adata[mask, :].copy()


def compute_distance_matrix(embedding: np.ndarray) -> np.ndarray:
    """
    Computes the pairwise distance matrix for a given embedding.

    Args:
        embedding: A 2D numpy array where rows represent points and columns represent dimensions.

    Returns:
        A 2D numpy array representing the pairwise distance matrix.
    """
    return squareform(pdist(embedding, 'euclidean'))

def compute_weight_matrix_from_distances(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Computes the weight matrix from a distance matrix using inverse distances.

    Args:
        distance_matrix: A 2D numpy array representing the pairwise distance matrix.

    Returns:
        A 2D numpy array representing the weight matrix.
    """
    with np.errstate(divide='ignore'):
        weight_matrix = 1 / distance_matrix
    np.fill_diagonal(weight_matrix, 0)  # weights for self-distance are set to zero
    return weight_matrix

def scale_values(values: np.ndarray, scaling_method: str, apply_log: bool, log_before_scaling: bool) -> np.ndarray:
    """
    Scales and optionally log-transforms the values.

    Args:
        values: The values to scale.
        scaling_method: The method to use for scaling ('minmax', 'standard', 'robust', or None).
        apply_log: Whether to apply log transformation.
        log_before_scaling: Whether to apply log transformation before scaling.

    Returns:
        The scaled (and optionally log-transformed) values.
    """
    if apply_log and log_before_scaling:
        values = np.log1p(values)

    if scaling_method == 'minmax':
        scaler = MinMaxScaler()
    elif scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'robust':
        scaler = RobustScaler()
    else:
        scaler = None

    if scaler is not None:
        values = scaler.fit_transform(values.reshape(-1, 1)).flatten()

    if apply_log and not log_before_scaling:
        values = np.log1p(values)

    return values

def neighbor_compute_moran_i(sub_adata: AnnData, value_key: str, category: str, use_embedding: bool = False, embedding_key: str = None, scaling_method: str = None, apply_log: bool = False, log_before_scaling: bool = False) -> dict:
    """
    Computes Moran's I spatial autocorrelation for a subset of cells.

    Args:
        sub_adata: Subset of AnnData object for specific cell type.
        value_key: The key in `sub_adata.obs` containing the values to analyze.
        category: The categorical variable in `sub_adata.obs` to group by.
        use_embedding: Whether to use embedding for distance calculation.
        embedding_key: The key in `sub_adata.obsm` for the embedding.
        scaling_method: The method to use for scaling ('minmax', 'standard', 'robust', or None).
        apply_log: Whether to apply log transformation.
        log_before_scaling: Whether to apply log transformation before scaling.

    Returns:
        A dictionary with Moran's I results.
    """
    try:
        if use_embedding and embedding_key is not None:
            embedding = sub_adata.obsm[embedding_key]
            distance_matrix = compute_distance_matrix(embedding)
            weight_matrix = compute_weight_matrix_from_distances(distance_matrix)
            sparse_weight_matrix = csr_matrix(weight_matrix)
            weights = WSP(sparse_weight_matrix)
        else:
            connectivities = sub_adata.obsp['connectivities']
            weights = WSP(connectivities)
    except KeyError as e:
        raise KeyError(f"Key error: {e}")

    values = sub_adata.obs[value_key].values
    values = scale_values(values, scaling_method, apply_log, log_before_scaling)
    weights_full = weights.to_W()
    moran = Moran(values, weights_full)
    return {
        category: sub_adata.obs[category].unique()[0],
        "Moran's I": moran.I,
        "P-value": moran.p_norm,
        "num_cell": len(values)
    }

def neighbor_process_cell_type(adata: AnnData, cell_type: str, value_key: str, category: str, use_embedding: bool = False, embedding_key: str = None, scaling_method: str = None, apply_log: bool = False, log_before_scaling: bool = False) -> pd.DataFrame:
    """
    Processes a specific cell type to compute Moran's I.

    Args:
        adata: The AnnData object to analyze.
        cell_type: The specific cell type to process.
        value_key: The key in `adata.obs` containing the values to analyze.
        category: The categorical variable in `adata.obs` to group by.
        use_embedding: Whether to use embedding for distance calculation.
        embedding_key: The key in `adata.obsm` for the embedding.
        scaling_method: The method to use for scaling ('minmax', 'standard', 'robust', or None).
        apply_log: Whether to apply log transformation.
        log_before_scaling: Whether to apply log transformation before scaling.

    Returns:
        A DataFrame with Moran's I results for the specific cell type.
    """
    mask = adata.obs[category] == cell_type
    num_cell = sum(mask)
    if num_cell > 10:
        sub_adata = adata[mask].copy()
        moranI_data = neighbor_compute_moran_i(sub_adata, value_key, category, use_embedding, embedding_key, scaling_method, apply_log, log_before_scaling)
        return pd.DataFrame([moranI_data])
    return pd.DataFrame()





def process_cell_type(cell_type, adata, value_key, category, use_embedding, embedding_key, scaling_method, apply_log, log_before_scaling):
    """
    Processes a specific cell type to compute Moran's I.

    Args:
        adata: The AnnData object to analyze.
        cell_type: The specific cell type to process.
        value_key: The key in `adata.obs` containing the values to analyze.
        category: The categorical variable in `adata.obs` to group by.
        use_embedding: Whether to use embedding for distance calculation.
        embedding_key: The key in `adata.obsm` for the embedding.
        scaling_method: The method to use for scaling ('minmax', 'standard', 'robust', or None).
        apply_log: Whether to apply log transformation.
        log_before_scaling: Whether to apply log transformation before scaling.

    Returns:
        A DataFrame with Moran's I results for the specific cell type.
    """
    logging.info(f"Processing cell type: {cell_type}")
    mask = adata.obs[category] == cell_type
    num_cell = sum(mask)
    if num_cell > 10:
        sub_adata = adata[mask].copy()
        moranI_data = neighbor_compute_moran_i(sub_adata, value_key, category, use_embedding, embedding_key, scaling_method, apply_log, log_before_scaling)
        return pd.DataFrame([moranI_data])
    else:
        logging.warning(f"No data available for cell type: {cell_type} (insufficient number of cells or other issues).")
        return pd.DataFrame()

def compute_neighbor_moran_i_by_category(
    adata: AnnData, 
    value_key: str, 
    category: str = "celltype", 
    connectivity_key: str = 'connectivities',
    use_embedding: bool = False,
    embedding_key: str = None,
    scaling_method: str = None,
    apply_log: bool = False,
    log_before_scaling: bool = False,
    max_workers: int = None,
    specific_celltype: list = None  # New argument to specify a single cell type
) -> pd.DataFrame:
    """
    Computes Moran's I spatial autocorrelation for each cell type, or a specific cell type, in parallel.

    Args:
        adata: The AnnData object to analyze.
        value_key: The key in `adata.obs` containing the values to analyze.
        category: The categorical variable in `adata.obs` to group by.
        connectivity_key: The key in `adata.obsp` containing the connectivities matrix.
        use_embedding: Whether to use embedding for distance calculation.
        embedding_key: The key in `adata.obsm` for the embedding.
        scaling_method: The method to use for scaling ('minmax', 'standard', 'robust', or None).
        apply_log: Whether to apply log transformation.
        log_before_scaling: Whether to apply log transformation before scaling.
        max_workers: The maximum number of threads to use for parallel processing.
        specific_celltype: A specific cell type to analyze. If provided, only this cell type will be analyzed.

    Returns:
        A DataFrame with Moran's I results for each cell type, or the specific cell type.
    """
    logging.info(f"Starting Moran's I computation with value key '{value_key}'.")

    # Determine the cell types to analyze
    if specific_celltype:
        logging.info(f"Analyzing specific cell type: {specific_celltype}")
        top_level_types = specific_celltype
    else:
        top_level_types = adata.obs[category].unique()
        logging.info(f"Identified {len(top_level_types)} unique cell types in category '{category}'.")

    result_df = pd.DataFrame()

    # Define a function to process each cell type, to be used with the thread pool
    def process_cell_type(cell_type):
        logging.info(f"Processing cell type: {cell_type}")
        cell_type_df = neighbor_process_cell_type(
            adata, cell_type, value_key, category, 
            use_embedding, embedding_key, scaling_method, apply_log, log_before_scaling
        )
        if cell_type_df.empty:
            logging.warning(f"No data available for cell type: {cell_type} (insufficient number of cells or other issues).")
        return cell_type_df

    # Use ThreadPoolExecutor to parallelize the execution if analyzing multiple cell types
    if len(top_level_types) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_cell_type, top_level_types))
    else:
        # If only one cell type is being analyzed, process it directly without parallelization
        results = [process_cell_type(top_level_types[0])]

    # Concatenate all results
    for cell_type_df in results:
        if not cell_type_df.empty:
            result_df = pd.concat([result_df, cell_type_df])

    if result_df.empty:
        logging.warning("No data was processed successfully. Returning an empty DataFrame.")
        return pd.DataFrame()

    result_df = result_df.set_index(category)
    result_df = result_df.sort_values("Moran's I", ascending=False)
    logging.info("Completed Moran's I computation.")

    return result_df


import concurrent.futures
import logging
import pandas as pd
import numpy as np

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_gene_expression(
    adatas: dict, 
    gene_list: list
) -> tuple[dict, dict]:
    """
    Extracts the gene expression data for the specified genes from a collection of AnnData objects.

    Parameters:
    -----------
    adatas : dict
        A dictionary where keys are identifiers and values are AnnData objects containing single-cell RNA-seq data.
    gene_list : list
        A list of gene names to extract from each AnnData object.

    Returns:
    --------
    tuple[dict, dict]:
        - A dictionary with the same keys as the input adatas, where the values are DataFrames containing the gene expression data.
        - A dictionary of any errors encountered during processing, with keys corresponding to the adatas keys and values as exceptions.
    """
    gene_expression_dict = {}
    processing_errors = {}

    def process_adata(key: str) -> tuple[str, pd.DataFrame]:
        """
        Processes a single AnnData object to extract the gene expression data.

        Parameters:
        -----------
        key : str
            The key identifying the AnnData object within the adatas dictionary.

        Returns:
        --------
        tuple[str, pd.DataFrame]:
            - The key corresponding to the AnnData object.
            - A DataFrame containing the gene expression data for the specified genes, with missing genes filled with None.
        """
        try:
            logging.info(f"Processing AnnData for key: {key}")
            
            # Create a mask for genes that exist in the AnnData object's var_names
            gene_exists_mask = [gene in adatas[key].var_names for gene in gene_list]
            existing_genes = [gene for gene, exists in zip(gene_list, gene_exists_mask) if exists]
            
            # Initialize a DataFrame with None for all columns
            expression_df = pd.DataFrame(None, index=adatas[key].obs_names, columns=gene_list)
            
            if existing_genes:
                # Extract expression data for the existing genes
                expression_data = adatas[key][:, existing_genes].X.toarray()
                expression_df.loc[:, existing_genes] = expression_data  # Assign the data to the corresponding columns
            
            logging.info(f"Successfully processed AnnData for key: {key}")
            return (key, expression_df)
        
        except Exception as exc:
            logging.error(f"Error processing AnnData for key: {key} - {exc}")
            return (key, exc)

    # Use ThreadPoolExecutor to process each AnnData object in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_adata, key): key for key in adatas.keys()}
        for future in concurrent.futures.as_completed(futures):
            key = futures[future]
            try:
                result_key, result_df = future.result()
                if isinstance(result_df, Exception):
                    processing_errors[result_key] = result_df
                else:
                    gene_expression_dict[result_key] = result_df
            except Exception as exc:
                processing_errors[key] = exc

    return gene_expression_dict, processing_errors




def identify_nearby_cells(
    merge_adata: ad.AnnData,
    label_column: str = 'celltype',
    target_col: str = 'datatype',
    targed_label: str = 'sn',
    threshold: float = 0.1,
    new_label_col: str = 'nearby_label'
) -> ad.AnnData:
    """
    Identifies and labels nearby cells in an AnnData object based on a threshold in the neighbor graph.

    Args:
        merge_adata: The AnnData object containing cells with connectivity information.
        label_column: The column name in `merge_adata.obs` which contains the labels for the target cells.
        target_col: The column name in `merge_adata.obs` representing labeled and unlabeled cells.
        targed_label: The label in `target_col` to consider as the source of nearby cells.
        threshold: The threshold for considering a cell "near" based on the neighbor graph connectivity.
        new_label_col: The column name to store new labels for nearby cells.

    Returns:
        A new AnnData object containing only the nearby cells that were identified.
    """
    import anndata as ad
    import numpy as np

    celltypes = merge_adata.obs[label_column].unique()
    merge_adata.obs[new_label_col] = 'unlabeled'  # Default value for cells that are not nearby
    nearby_cellbin_indices_dict = {}

    for celltype in celltypes:
        print(f"Processing celltype: {celltype}")
        sn_mask = (merge_adata.obs[target_col] == targed_label) & (merge_adata.obs[label_column] == celltype)
        cellbin_mask = (merge_adata.obs[target_col] != targed_label) & (merge_adata.obs[label_column] == celltype)
        neighbor_graph = merge_adata.obsp['connectivities']
        sn_indices = np.where(sn_mask)[0]
        cellbin_indices = np.where(cellbin_mask)[0]

        if len(sn_indices) == 0 or len(cellbin_indices) == 0:
            print(f"No labeled or unlabeled cells found for celltype: {celltype}")
            continue

        neighbor_sums = neighbor_graph[sn_indices].sum(axis=0)
        neighbor_sums = np.asarray(neighbor_sums).flatten()
        nearby_cellbin_indices = cellbin_indices[neighbor_sums[cellbin_indices] > threshold]

        if len(nearby_cellbin_indices) > 0:
            nearby_cellbin_indices_dict[celltype] = nearby_cellbin_indices
            merge_adata.obs.loc[merge_adata.obs.index[nearby_cellbin_indices], new_label_col] = f'near_{celltype}'
        else:
            print(f"No nearby unlabeled cells found for celltype: {celltype}")

    all_nearby_cellbin_indices = np.concatenate(list(nearby_cellbin_indices_dict.values()))
    print(f"Total nearby unlabeled cells found: {merge_adata[all_nearby_cellbin_indices].shape[0]}")

    return merge_adata[all_nearby_cellbin_indices].copy()




