import scanpy as sc

import cellbin_moran as cm

import anndata as ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

region = "PFC"

ref_pfc = cm.load_sct_and_set_index(f"/home1/jijh/st_project/cellbin_analysis/annotated_cell_bins/sn_sct_h5ad/{region}_sct_counts.h5ad")
ref_pfc_sub = cm.analysis.hierarchical_sample(ref_pfc, groupby_cols=["sample", "celltype"], n_samples=10000, random_state=1)
ref_pfc_sub = ref_pfc_sub.raw.to_adata()

ref_pfc_sub.obs["celltype"] = ref_pfc_sub.obs["fine"].str.split("-").str[0]
ref_pfc_sub.obs.groupby("celltype").count()

# Reusing the previously defined functions:
cellbin_dir = "/public/home/jijh/st_project/cellbin_analysis/annotated_cell_bins/sct_cellbin_h5ad"
meta_dir = "/public/home/jijh/st_project/cellbin_analysis/annotated_cell_bins/region_meta/"


metas = cm.io.read_and_process_metadata(meta_dir, criteria=f"'csv' in file")

sample = cm.list_files_matching_criteria(cellbin_dir, condition=f"'h5ad' in file")

cellbin_data = cm.load_data_in_parallel(sample, cm.load_sct_and_set_index)

for key in cellbin_data.keys():
    cellbin_data[key] = cellbin_data[key].raw.to_adata()
    cellbin_data[key].obs = metas[key]

for key, adata_sample in cellbin_data.items():
    print(f"Processing {key} sample")
    adata_pdf = cm.analysis.subset_anndata(adata_sample, {"Structure Name": "Prefrontal cortex"})
    adata_pdf.obs["datatype"] = "cellbin"
    merge_adata = cm.analysis.concatenate_and_intersect([adata_pdf, ref_pfc_sub])
    merge_adata.layers["counts"] = merge_adata.X
    merge_adata.obs["celltype"] = merge_adata.obs["fine"].str.split("-").str[0]
    
    sc.pp.normalize_total(merge_adata, target_sum=1e4)
    sc.pp.log1p(merge_adata)
    sc.pp.highly_variable_genes(merge_adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.pl.highly_variable_genes(merge_adata)
    merge_adata.raw = merge_adata
    merge_adata = merge_adata[:, merge_adata.var.highly_variable]
    sns.kdeplot(data = merge_adata.obs, x = "nCount_SCT", hue = "datatype")
    # Start regress
    sc.pp.regress_out(merge_adata, ["nCount_SCT"])
    sc.pp.scale(merge_adata, max_value=10)
    sc.tl.pca(merge_adata, svd_solver="arpack")
    sc.pl.pca(merge_adata, color="Thy1", title = key)
    # Raw UMAP
    sc.pp.neighbors(merge_adata, n_neighbors=10, n_pcs=40)
    sc.tl.umap(merge_adata)
    merge_adata.obsm["X_pca_umap"] = merge_adata.obsm["X_umap"]
    # Harmony
    sc.external.pp.harmony_integrate(merge_adata, key="datatype")
    sc.pp.neighbors(merge_adata, n_neighbors=10, n_pcs=40, use_rep="X_pca_harmony")
    sc.tl.umap(merge_adata)
    merge_adata.obsm["X_harmony_umap"] = merge_adata.obsm["X_umap"]
    sc.pl.umap(merge_adata, color = "celltype", ax = ax, palette=cm.palettes.general_type_colors)
    mask = (merge_adata.obs["celltype"] == 'Micro') & (merge_adata.obs["datatype"] == 'cellbin')
    sc.pl.umap(merge_adata[mask], color = "fine", palette=cm.palettes.cell_type_colors)
    mask = merge_adata.obs["datatype"] == 'cellbin'
    nei_df = cm.analysis.compute_neighbor_moran_i_by_category(merge_adata[mask], "min_center_dist")
    # Store the result
    nei_df.to_csv
    merge_adata.write(f"/public/home/jijh/st_project/cellbin_analysis/annotated_cell_bins/umap_harmony/{key}_umap_harmony.h5ad")












regions = { 'ENT': 'Entorhinal area', 'HPF': 'Hippocampal formation', 'STR': 'Striatum', 'TH': 'Thalamus', 'RS': 'Retrosplenial area', 'PFC': 'Prefrontal cortex', 'BF': 'Basal forebrain' }
for key, adata_sample in cellbin_data.items():
    print(f"Processing {key} sample")
    adata_pdf = cm.analysis.subset_anndata(adata_sample, {"Structure Name": regions[region]})
    adata_pdf.obs["datatype"] = "cellbin"
    merge_adata = cm.analysis.concatenate_and_intersect([adata_pdf, ref_pfc_sub])
    merge_adata.layers["counts"] = merge_adata.X
    merge_adata.obs["celltype"] = merge_adata.obs["fine"].str.split("-").str[0]
    
    sc.pp.normalize_total(merge_adata, target_sum=1e4)
    sc.pp.log1p(merge_adata)
    sc.pp.highly_variable_genes(merge_adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.pl.highly_variable_genes(merge_adata)
    merge_adata.raw = merge_adata
    merge_adata = merge_adata[:, merge_adata.var.highly_variable]
    sns.kdeplot(data = merge_adata.obs, x = "nCount_SCT", hue = "datatype")
    # Start regress
    sc.pp.regress_out(merge_adata, ["nCount_SCT"])
    sc.pp.scale(merge_adata, max_value=10)
    sc.tl.pca(merge_adata, svd_solver="arpack")
    sc.pl.pca(merge_adata, color="Thy1", title = key)
    # Raw UMAP
    sc.pp.neighbors(merge_adata, n_neighbors=10, n_pcs=40)
    sc.tl.umap(merge_adata)
    merge_adata.obsm["X_pca_umap"] = merge_adata.obsm["X_umap"]
    # Harmony
    sc.external.pp.harmony_integrate(merge_adata, key="datatype")
    sc.pp.neighbors(merge_adata, n_neighbors=10, n_pcs=40, use_rep="X_pca_harmony")
    sc.tl.umap(merge_adata)
    merge_adata.obsm["X_harmony_umap"] = merge_adata.obsm["X_umap"]
    sc.pl.umap(merge_adata, color = "celltype", ax = ax, palette=cm.palettes.general_type_colors)
    mask = (merge_adata.obs["celltype"] == 'Micro') & (merge_adata.obs["datatype"] == 'cellbin')
    sc.pl.umap(merge_adata[mask], color = "fine", palette=cm.palettes.cell_type_colors)
    mask = merge_adata.obs["datatype"] == 'cellbin'
    nei_df = cm.analysis.compute_neighbor_moran_i_by_category(merge_adata[mask], "min_center_dist")
    # Store the result
    nei_df.to_csv(f"/home1/jijh/st_project/cellbin_analysis/annotated_cell_bins/regress_harmony/{key}_{region}_regress_moranI.csv")
    merge_adata.write(f"/home1/jijh/st_project/cellbin_analysis/annotated_cell_bins/regress_harmony/{key}_{region}_regress_harmony.h5ad")