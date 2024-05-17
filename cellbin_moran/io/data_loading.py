import scanpy as sc

def load_sct_and_set_index(adata_path):
    adata = sc.read_h5ad(adata_path)
    adata.raw.var.set_index("_index", inplace=True)
    adata.var.set_index("_index", inplace=True)
    return adata
