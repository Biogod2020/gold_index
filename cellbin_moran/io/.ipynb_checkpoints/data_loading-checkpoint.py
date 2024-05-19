# your_project/io/data_loading.py

import scanpy as sc

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
