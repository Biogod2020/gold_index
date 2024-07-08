library(Seurat)
regions = c("BF", "ENT", "HPF", "PFC", "RS", "STR", "TH")
celltypes = c("ExN", "InN")
for (i in 1:length(regions)){
sce = readRDS(paste0("regions[i],"_SN_all_annot.rds"))
for (j in 1:length(celltypes)){
Idents(sce) = as.factor(sce$celltype)
sce1 = subset(sce, idents = celltypes[j])
sce1 <- NormalizeData(sce1, normalization.method = "LogNormalize", scale.factor = 10000)
sce1 <- FindVariableFeatures(sce1, selection.method = "vst", nfeatures = 3000)
sce1 <- ScaleData(sce1)
sce1 <- RunPCA(sce1, features = VariableFeatures(object = sce1))
sce1 <- RunUMAP(sce1, dims = 1:10)
}