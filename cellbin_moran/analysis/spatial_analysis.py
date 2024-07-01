# your_project/analysis/spatial_analysis.py

import pandas as pd
import squidpy as sq
import numpy as np
import scanpy as sc
import anndata as ad
from sklearn.utils import check_random_state
from scipy.sparse import csr_matrix
from libpysal.weights import WSP
from esda.moran import Moran
from typing import List, Dict
from anndata import AnnData
import concurrent.futures
import warnings

def compute_moranI(
    adata: ad.AnnData, 
    genes: list[str], 
    category: str = "celltype", 
    spatial_key: str = "X_umap"
) -> pd.DataFrame:
    """
    Computes Moran's I spatial autocorrelation for specified genes.

    Args:
        adata: The AnnData object to analyze.
        genes: List of gene names to compute Moran's I for.
        category: The categorical variable in `adata.obs` to group by.
        spatial_key: The key in `adata.obsm` containing spatial coordinates.

    Returns:
        A DataFrame with Moran's I results for each cell type and gene.
    """
    top_level_types = adata.obs[category].unique()
    result_df = pd.DataFrame()
    data_present = False
    
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
            data_present = True
    
    if not data_present:
        return pd.DataFrame()
    
    result_df = result_df.set_index(category)
    result_df = result_df.sort_values("I", ascending=False)
    return result_df

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


def neighbor_normalize_and_log_transform(adata: AnnData, value_key: str) -> pd.Series:
    """
    Scales the `value_key` column in `adata.obs` by its min and max values, normalizes it to 1e5, and log10 transforms it.

    Args:
        adata: The AnnData object containing the data.
        value_key: The key in `adata.obs` containing the values to scale, normalize, and transform.
    """
    values = adata.obs[value_key]
    min_val = values.min()
    max_val = values.max()
    
    # Scale the data
    scaled_values = (values - min_val) / (max_val - min_val)
    
    # Normalize to 1e5 and log10 transform
    normalized_transformed_values = np.log10((scaled_values * 1e5) + 1)
    
    return normalized_transformed_values


def neighbor_compute_moran_i(sub_adata: AnnData, value_key: str, category: str) -> dict:
    """
    Computes Moran's I spatial autocorrelation for a subset of cells.

    Args:
        sub_adata: Subset of AnnData object for specific cell type.
        value_key: The key in `sub_adata.obs` containing the values to analyze.
        category: The categorical variable in `sub_adata.obs` to group by.

    Returns:
        A dictionary with Moran's I results.
    """
    try:
        connectivities = sub_adata.obsp['connectivities']
    except KeyError:
        raise KeyError(f"Connectivity key 'connectivities' not found in sub_adata.obsp")

    weights = WSP(connectivities)
    values = sub_adata.obs[value_key].values
    weights_full = weights.to_W()
    moran = Moran(values, weights_full)
    return {
        category: sub_adata.obs[category].unique()[0],
        "Moran's I": moran.I,
        "P-value": moran.p_norm,
        "num_cell": len(values)
    }

def neighbor_process_cell_type(adata: AnnData, cell_type: str, value_key: str, category: str) -> pd.DataFrame:
    """
    Processes a specific cell type to compute Moran's I.

    Args:
        adata: The AnnData object to analyze.
        cell_type: The specific cell type to process.
        value_key: The key in `adata.obs` containing the values to analyze.
        category: The categorical variable in `adata.obs` to group by.

    Returns:
        A DataFrame with Moran's I results for the specific cell type.
    """
    mask = adata.obs[category] == cell_type
    num_cell = sum(mask)
    if num_cell > 10:
        sub_adata = adata[mask].copy()
        moranI_data = neighbor_compute_moran_i(sub_adata, value_key, category)
        return pd.DataFrame([moranI_data])
    return pd.DataFrame()

def compute_neighbor_moran_i_by_category(
    adata: AnnData, 
    value_key: str, 
    category: str = "celltype", 
    connectivity_key: str = 'connectivities'
) -> pd.DataFrame:
    """
    Computes Moran's I spatial autocorrelation for each cell type.

    Args:
        adata: The AnnData object to analyze.
        value_key: The key in `adata.obs` containing the values to analyze.
        category: The categorical variable in `adata.obs` to group by.
        connectivity_key: The key in `adata.obsp` containing the connectivities matrix.

    Returns:
        A DataFrame with Moran's I results for each cell type.
    """
    top_level_types = adata.obs[category].unique()
    result_df = pd.DataFrame()
    data_present = False
    
    for cell_type in top_level_types:
        cell_type_df = neighbor_process_cell_type(adata, cell_type, value_key, category)
        if not cell_type_df.empty:
            result_df = pd.concat([result_df, cell_type_df])
            data_present = True
    
    if not data_present:
        return pd.DataFrame()
    
    result_df = result_df.set_index(category)
    result_df = result_df.sort_values("Moran's I", ascending=False)
    return result_df

def process_anndata_compute_neighbor_moran_i(
    adata_path: str, 
    value_key: str, 
    category: str, 
    connectivity_key: str,
    normalize_log: bool
) -> pd.DataFrame:
    """
    Processes an AnnData object from a file path to compute Moran's I.

    Args:
        adata_path: The file path to the AnnData object.
        value_key: The key in `adata.obs` containing the values to analyze.
        category: The categorical variable in `adata.obs` to group by.
        connectivity_key: The key in `adata.obsp` containing the connectivities matrix.

    Returns:
        A DataFrame with Moran's I results for each cell type.
    """
    adata = sc.read_h5ad(adata_path)
    if normalize_log == True:
        adata.obs[value_key] = neighbor_normalize_and_log_transform(adata, value_key)
        print(adata.obs[value_key].describe())
        
    return compute_neighbor_moran_i_by_category(adata, value_key, category, connectivity_key)