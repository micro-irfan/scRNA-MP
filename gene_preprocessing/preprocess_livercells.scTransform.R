suppressPackageStartupMessages({
  library(Matrix)
  library(Seurat)
  library(SeuratDisk)  # install.packages("SeuratDisk")
})

# ---- your mapping (same content as python) ----
batch_dict <- list(
  DM_PIP_3in4  = c("RHK516","RHK518"),
  DMS_PIP_3in4 = c("RHK517","RHK519"),
  DM_PIP_1in4  = c("RHK520","RHK522"),
  DMS_PIP_1in4 = c("RHK521","RHK523")
)
treatment_order <- c("DMSO","NAIN3")  # assumes [1]=DMSO, [2]=NAIN3

make_batch_reverse <- function(batch_dict, treatment_order = c("DMSO","NAIN3")) {
  rev <- list()
  for (k in names(batch_dict)) {
    v <- batch_dict[[k]]
    if (length(v) != 2) stop("batch_dict['", k, "'] must have exactly 2 entries.")
    rev[[v[[1]]]] <- paste0(k, "_", treatment_order[[1]])
    rev[[v[[2]]]] <- paste0(k, "_", treatment_order[[2]])
  }
  rev
}
batch_dict_reverse <- make_batch_reverse(batch_dict, treatment_order)

# ---- helpers ----

# merge two genes×cells matrices by union genes; sum overlaps
merge_gene_and_aux <- function(gene_mat, aux_mat) {
  stopifnot(inherits(gene_mat, "sparseMatrix"), inherits(aux_mat, "sparseMatrix"))

  common <- intersect(colnames(gene_mat), colnames(aux_mat))
  stopifnot(length(common) > 0)

  gene_mat <- gene_mat[, common, drop = FALSE]
  aux_mat  <- aux_mat[,  common, drop = FALSE]

  allg <- union(rownames(gene_mat), rownames(aux_mat))

  # Create 0-filled aligned matrices by name (no NA indexing)
  g <- Matrix::Matrix(0, nrow = length(allg), ncol = length(common), sparse = TRUE,
                      dimnames = list(allg, common))
  a <- Matrix::Matrix(0, nrow = length(allg), ncol = length(common), sparse = TRUE,
                      dimnames = list(allg, common))

  g[rownames(gene_mat), ] <- gene_mat
  a[rownames(aux_mat),  ] <- aux_mat

  # Merge by summing (simpler + faster than rbind+rowsum)
  merged <- g + a
  merged
}


# combine many samples: union genes, concat cells
combine_samples <- function(mats) {
  stopifnot(length(mats) > 0)

  allg <- Reduce(union, lapply(mats, rownames))

  mats2 <- lapply(mats, function(m) {
    # Make a 0-filled matrix with the target rownames and the same columns as m
    mm <- Matrix::Matrix(0,
                         nrow = length(allg),
                         ncol = ncol(m),
                         sparse = TRUE,
                         dimnames = list(allg, colnames(m)))

    # Fill the rows that exist in m
    mm[rownames(m), ] <- m
    mm
  })

  out <- mats2[[1]]
  if (length(mats2) > 1) {
    for (i in 2:length(mats2)) out <- Matrix::cbind2(out, mats2[[i]])
  }
  out
}


add_metadata_from_barcodes <- function(obj) {
  batch <- colnames(obj) 

  obj[["batch"]] <- batch
  obj[["treatment"]] <- ifelse(grepl("DMSO", batch), "DMSO", "NAIN3")
  obj[["steatosis"]] <- ifelse(grepl("^DM_", batch), "DM", "DMS")
  obj[["dilution"]] <- ifelse(grepl("1in4", batch), "1in4", "3in4")

  obj
}

`%||%` <- function(a, b) if (!is.null(a)) a else b

run_sct_umap_leiden <- function(counts,
                                mt_pattern = "^MT-",
                                ribo_pattern = "^(RPS|RPL)",
                                vars_to_regress = c("percent.mt","percent.ribo"),
                                variable_features = 3000,
                                pca_npcs = 50,
                                use_pcs = 30,
                                resolution = 0.5,
                                seed = 1) {
  set.seed(seed)

  obj <- CreateSeuratObject(counts = counts, assay = "RNA")

  # QC percentages from gene symbols
  obj <- PercentageFeatureSet(obj, pattern = "^MT-", col.name = "percent.mt")
  obj <- PercentageFeatureSet(obj, pattern = "^HB[^(P)]", col.name = "percent.hb")
  obj <- PercentageFeatureSet(obj, pattern = "^RPS|^RP", col.name = "percent.ribo")

  # obj <- subset(
  #   obj,
  #   subset =
  #     nFeature_RNA >= 500 &     # detected genes
  #     nCount_RNA >= 1000 &      # total UMIs
  #     percent.mt <= 20          # mitochondrial % 
  # )
  
  # Gene QC by number of cells : standard 10
  counts <- GetAssayData(obj, assay = "RNA", slot = "counts")
  keep_genes <- rowSums(counts > 0) >= 10
  obj <- obj[keep_genes, ]

  exclude = c("percent.hb", "percent.ribo")

  # Only keep regressors that exist
  vars_to_regress <- vars_to_regress[
    vars_to_regress %in% colnames(obj@meta.data) &
    !(vars_to_regress %in% exclude)
  ]

  if (length(vars_to_regress) == 0) vars_to_regress <- NULL

  # vars_to_regress <- NULL

  obj <- SCTransform(
    obj,
    assay = "RNA",
    new.assay.name = "SCT",
    variable.features.n = variable_features,
    vars.to.regress = vars_to_regress,
    verbose = TRUE
  )

  obj <- SCTransform(obj)
  DefaultAssay(obj) <- "SCT"

  ## Added Cell Cycle Scoring
  obj <- CellCycleScoring(
    obj,
    s.features   = cc.genes$s.genes,
    g2m.features = cc.genes$g2m.genes,
    set.ident    = FALSE
  )

  obj <- RunPCA(obj, assay = "SCT", npcs = pca_npcs, verbose = TRUE)
  use_pcs <- min(use_pcs, pca_npcs, ncol(Embeddings(obj, "pca")))

  obj <- FindNeighbors(obj, dims = 1:use_pcs, verbose = TRUE)

  options(scipen = 999)
  obj <- FindClusters(obj, resolution = resolution, algorithm = 1, verbose = TRUE) # Leiden
  obj <- RunUMAP(obj, dims = 1:use_pcs, verbose = TRUE)

  obj
}

export_to_h5ad <- function(obj, out_h5ad_path, obj_to_save) {
  # writes .h5seurat then converts
  h5s <- sub("\\.h5ad$", ".h5seurat", out_h5ad_path)

  DefaultAssay(obj) <- obj_to_save

  SaveH5Seurat(obj, filename = h5s, overwrite = TRUE)
  Convert(h5s, dest = "h5ad", overwrite = TRUE)
  invisible(out_h5ad_path)
}

guess_sep <- function(path) {
  ext <- tolower(tools::file_ext(path))
  if (ext %in% c("tsv", "txt")) return("\t")
  if (ext %in% c("csv")) return(",")
  line1 <- readLines(path, n = 1, warn = FALSE)
  if (grepl("\t", line1)) return("\t")
  if (grepl(",", line1)) return(",")
  return("\t")  # safe default for count matrices
}

read_count_matrix <- function(path, sep = NULL, gene_col = 1L, collapse_dups = TRUE) {
  if (is.null(sep)) sep <- guess_sep(path)
  dt <- data.table::fread(
    path,
    sep = sep,
    header = TRUE,
    data.table = FALSE,
    check.names = FALSE
  )

  # 1. Store the names before altering the table
  cell_names <- dt[[1]]
  genes <- colnames(dt)[-1] # Exclude the ID column name

  # 2. Transpose EVERYTHING EXCEPT the first column
  dt_t <- data.table::transpose(dt[, -1])

  # 3. Assign names
  colnames(dt_t) <- cell_names
  rownames(dt_t) <- genes

  # 4. Final data frame
  dt <- as.data.frame(dt_t)

  if (ncol(dt) < 2) stop("Input file has <2 columns: ", path)

  # genes <- as.character(dt[[gene_col]])
  # dt[[gene_col]] <- NULL

  mat <- as.matrix(dt)
  rownames(mat) <- genes

  suppressWarnings(storage.mode(mat) <- "numeric")
  mat[is.na(mat)] <- 0

  # Handle duplicate gene symbols: sum counts across duplicates
  if (collapse_dups && any(duplicated(rownames(mat)))) {
    mat <- rowsum(mat, group = rownames(mat), reorder = FALSE)
  } else {
    rownames(mat) <- make.unique(rownames(mat))
  }

  # Convert to sparse where possible
  Matrix::Matrix(mat, sparse = TRUE)
}



get_gene_matrix <- function(sample_id) {
  # count_location <- '/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/cellline/results_3/matrices/bam_gene_count'
  # gene_path <- file.path(count_location, sample_id, "gene_count.mx")

  # count_location <- '/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/cellline/data_3'
  # gene_path <- file.path(count_location, sample_id, "expression",
  #                        paste0(sample_id, "_filter40_exp_umi.tsv"))

  count_location <- '/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/cellline/results_3/fastp_filter/preprocessing'
  gene_path <- file.path(count_location, sample_id, "expression",
                         paste0(sample_id, "_filter40_exp_fastp.tsv"))
  
  if (!file.exists(gene_path)) stop("Missing gene count file: ", gene_path)

  read_count_matrix(gene_path)
}



pc_var_explained <- function(seu) {
  sdev <- seu[["pca"]]@stdev
  v <- (sdev^2) / sum(sdev^2)
  v
}

rename_cells <- function(mat, sample_id) {
  colnames(mat) <- paste0(sample_id, "-bc", seq_len(ncol(mat)))
  mat
}

# ---- main driver  ----
threeInFour <- c("DMS_PIP_DMSO_3in4","DMS_PIP_NAIN3_3in4","DM_PIP_DMSO_3in4","DM_PIP_NAIN3_3in4")
oneInFour   <- c("DMS_PIP_DMSO_1in4","DMS_PIP_NAIN3_1in4","DM_PIP_DMSO_1in4","DM_PIP_NAIN3_1in4")

sample_dict <- list(
  combined = c(threeInFour, oneInFour)
  # combined = c("DMS_PIP_DMSO_3in4", "DM_PIP_DMSO_3in4", "DMS_PIP_DMSO_1in4", "DM_PIP_DMSO_1in4")
  # combined = c("DMS_PIP_NAIN3_3in4", "DM_PIP_NAIN3_3in4", "DMS_PIP_NAIN3_1in4", "DM_PIP_NAIN3_1in4")
)

for (analysis in names(sample_dict)) {
  sample_list <- sample_dict[[analysis]]

  per_sample <- lapply(sample_list, function(sid) {
    gmat <- get_gene_matrix(sid)
    rename_cells(gmat, sid)
  })

  names(per_sample) <- sample_list

  # across-samples: combine
  counts <- combine_samples(per_sample)

  # build Seurat + metadata + SCT + PCA/UMAP/Leiden
  obj <- CreateSeuratObject(counts = counts, assay = "RNA")

  obj <- run_sct_umap_leiden(
    counts = GetAssayData(obj, assay="RNA", slot="counts"),
    mt_pattern = "^MT-",
    ribo_pattern = "^(RPS|RPL)",
    vars_to_regress = c("percent.mt","percent.ribo"), 
    variable_features = 3000,
    pca_npcs = 50,
    use_pcs = 30,
    resolution = 0.5,
    seed = 1
  )

  # Exclude percent ribo 
  # Remove <10% ribo for next analysis - huge gap between the high end (20%) and lower end (0)

  # Variation to Regress

  obj <- add_metadata_from_barcodes(obj)

  out <- sprintf("fastp_filterNone/adata_processed.%s.scT.RNA.h5ad", analysis)
  export_to_h5ad(obj, out, "RNA")

  out <- sprintf("fastp_filterNone/adata_processed.%s.scT.SCT.h5ad", analysis)
  export_to_h5ad(obj, out, "SCT")

  message("Wrote: ", out)

  writeMM(GetAssayData(obj, assay="RNA", slot="counts"), "fastp_filterNone/counts.mtx")
  writeMM(GetAssayData(obj, assay="SCT", slot="data"),  "fastp_filterNone/sct.mtx")
  
  sct_counts <- GetAssayData(
    obj,
    assay = "SCT",
    slot  = "counts"
  )

  writeMM(sct_counts, "fastp_filterNone/SCT_corrected_counts.mtx")
  write.table(
    rownames(sct_counts),
    "fastp_filterNone/SCT_genes.tsv",
    quote = FALSE,
    row.names = FALSE,
    col.names = FALSE
  )
  write.table(
    colnames(sct_counts),
    "fastp_filterNone/SCT_barcodes.tsv",
    quote = FALSE,
    row.names = FALSE,
    col.names = FALSE
  )

  # ---------- outputs ----------
  outdir <- "/home/users/astar/gis/muhdih/scratch/sgRNA_mutational_rate/cellline/src/9_Rscripts_support/fastp_filterNone"
  rds_path <- file.path(outdir, paste0(analysis, "_SCT_PCA_UMAP_Leiden.rds"))
  saveRDS(obj, rds_path)

  # Save PC variance explained
  pvar <- pc_var_explained(obj)
  pvar_df <- data.frame(PC = paste0("PC", seq_along(pvar)),
                        variance = pvar,
                        variance_pct = 100 * pvar,
                        cumulative_pct = 100 * cumsum(pvar))
  write.csv(pvar_df, file.path(outdir, paste0(analysis, "_pca_variance.csv")), row.names = FALSE)

  meta <- obj@meta.data
  meta$cell_id <- rownames(meta)
  keep_cols <- intersect(c("cell_id", "batch", "nCount_RNA", "nFeature_RNA", "percent.mt", "seurat_clusters"), colnames(meta))
  write.csv(meta[, keep_cols, drop = FALSE],
            file.path(outdir, paste0(analysis, "_cell_metadata_clusters.csv")),
            row.names = FALSE)

  pca_emb <- Embeddings(obj, "pca")
  write.csv(cbind(cell_id = rownames(pca_emb), pca_emb),
            file.path(outdir, paste0(analysis, "_pca_embeddings.csv")),
            row.names = FALSE)

  umap_emb <- Embeddings(obj, "umap")
  write.csv(cbind(cell_id = rownames(umap_emb), umap_emb),
            file.path(outdir, paste0(analysis, "_umap_embeddings.csv")),
            row.names = FALSE)

}
