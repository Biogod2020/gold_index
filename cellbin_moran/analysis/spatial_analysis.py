# your_project/analysis/spatial_analysis.py

import pandas as pd
import squidpy as sq
import numpy as np
import scanpy as sc
import anndata as ad
from sklearn.utils import check_random_state

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
    from sklearn.utils import check_random_state

    if (n_samples is not None and fraction is not None) or (n_samples is None and fraction is None):
        raise ValueError("Must specify exactly one of 'n_samples' or 'fraction'.")

    rng = check_random_state(random_state)

    def _sample_group(group: pd.DataFrame, level: int) -> pd.DataFrame:
        """Recursively samples from each group."""
        if level == len(groupby_cols) - 1:
            if fraction is not None:
                return group.sample(frac=fraction, random_state=rng)
            elif isinstance(n_samples, int):
                return group.sample(n=n_samples, random_state=rng)
            elif isinstance(n_samples, float):
                return group.sample(frac=n_samples, random_state=rng)
        else:
            return group.groupby(groupby_cols[level + 1]).apply(lambda x: _sample_group(x, level + 1))

    sampled_obs = _sample_group(adata.obs, level=0)
    return adata[sampled_obs.index, :].copy()
