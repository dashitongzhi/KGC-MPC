"""Data loaders for Paper 3 (KGC-MPC): Norman 2019, Adamson 2016, Sci-Plex3 (GSE139944), and Sci-Plex4."""
from __future__ import annotations

import gzip
import os
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


NORMAN_URL = "https://dataverse.harvard.edu/api/access/datafile/6154020"
ADAMSON_URL = "https://dataverse.harvard.edu/api/access/datafile/6154417"

SCIPLEX3_URLS = {
    "matrix": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4150nnn/GSM4150378/suppl/GSM4150378_sciPlex3_A549_MCF7_K562_screen_UMI.count.matrix.gz",
    "genes": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4150nnn/GSM4150378/suppl/GSM4150378_sciPlex3_A549_MCF7_K562_screen_gene.annotations.txt.gz",
    "cells": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4150nnn/GSM4150378/suppl/GSM4150378_sciPlex3_A549_MCF7_K562_screen_cell.annotations.txt.gz",
    "pdata": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4150nnn/GSM4150378/suppl/GSM4150378_sciPlex3_pData.txt.gz",
}

SCIPLEX4_URLS = {
    "matrix": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4150nnn/GSM4150379/suppl/GSM4150379_sciPlex4_A549_MCF7_HDACi_UMI.count.matrix.gz",
    "genes": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4150nnn/GSM4150379/suppl/GSM4150379_sciPlex4_A549_MCF7_HDACi_gene.annotations.txt.gz",
    "cells": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4150nnn/GSM4150379/suppl/GSM4150379_sciPlex4_A549_MCF7_HDACi_cell.annotations.txt.gz",
    "pdata": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4150nnn/GSM4150379/suppl/GSM4150379_sciPlex4_pData.txt.gz",
}


def download_file(url: str, dest: Path, min_bytes: int = 1000) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return dest
    subprocess.run(
        [
            "curl",
            "-fL",
            "--retry",
            "5",
            "--retry-delay",
            "3",
            "-A",
            "Mozilla/5.0",
            "-o",
            str(dest),
            url,
        ],
        check=True,
    )
    if dest.stat().st_size < min_bytes:
        raise RuntimeError(f"Downloaded file {dest} too small ({dest.stat().st_size} bytes)")
    return dest


def load_gears_perturb_h5ad(name: str, data_dir: Path) -> ad.AnnData:
    """Download and load Norman 2019 or Adamson 2016 harmonized Perturb-seq AnnData."""
    url = NORMAN_URL if name.lower() == "norman" else ADAMSON_URL
    h5_cache = data_dir / f"{name}_perturb_processed.h5ad"
    if h5_cache.exists() and h5_cache.stat().st_size > 1_000_000:
        return ad.read_h5ad(h5_cache)

    dl_path = data_dir / f"{name}_download.bin"
    download_file(url, dl_path, min_bytes=5_000_000)
    if zipfile.is_zipfile(dl_path):
        with zipfile.ZipFile(dl_path, "r") as zf:
            h5_members = [m for m in zf.namelist() if m.endswith(".h5ad")]
            if not h5_members:
                raise RuntimeError(f"No .h5ad inside {dl_path}: {zf.namelist()}")
            member = h5_members[0]
            with zf.open(member) as src, open(h5_cache, "wb") as dst:
                dst.write(src.read())
    else:
        dl_path.rename(h5_cache)
    return ad.read_h5ad(h5_cache)


def _load_sparse_triplet_matrix(
    f_mat: Path,
    n_cells_all: int,
    n_genes_all: int,
    chunksize: int = 20_000_000,
) -> sp.csr_matrix:
    """Stream-load gzipped 1-based triplet matrix using Pandas C-engine chunks to keep peak RAM low."""
    skiprows = 0
    delim = " "
    with gzip.open(f_mat, "rt", encoding="utf-8", errors="replace") as fh:
        for _ in range(10):
            line = fh.readline()
            if not line:
                break
            stripped = line.strip()
            if stripped.startswith("%"):
                skiprows += 1
                continue
            parts = stripped.split()
            if len(parts) == 3:
                try:
                    p0, p1 = int(parts[0]), int(parts[1])
                    if p0 == n_genes_all and p1 == n_cells_all:
                        skiprows += 1
                        continue
                except ValueError:
                    pass
            delim = "\t" if "\t" in line else " "
            break

    X_acc = sp.csr_matrix((n_cells_all, n_genes_all), dtype=np.float32)
    reader = pd.read_csv(
        f_mat,
        sep=delim,
        engine="c",
        skipinitialspace=(delim == " "),
        skiprows=skiprows,
        header=None,
        names=["g", "c", "v"],
        dtype={"g": np.int32, "c": np.int32, "v": np.float32},
        chunksize=chunksize,
    )
    for chunk in reader:
        rows = chunk["c"].to_numpy(dtype=np.int32, copy=False) - 1
        cols = chunk["g"].to_numpy(dtype=np.int32, copy=False) - 1
        vals = chunk["v"].to_numpy(dtype=np.float32, copy=False)
        X_acc = X_acc + sp.coo_matrix((vals, (rows, cols)), shape=(n_cells_all, n_genes_all), dtype=np.float32).tocsr()
    return X_acc


def _read_genes_df(f_genes: Path) -> pd.DataFrame:
    """Load Sci-Plex gene table whether headered ('id gene_short_name') or headerless."""
    with gzip.open(f_genes, "rt", encoding="utf-8", errors="replace") as fh:
        first_line = fh.readline().strip()
    if "gene_short_name" in first_line or first_line.lower().startswith("id "):
        return pd.read_csv(f_genes, sep=r"\s+")
    return pd.read_csv(f_genes, sep=r"\s+", header=None, names=["id", "gene_short_name"])


def load_sciplex_matrix_to_hvg(
    urls: Dict[str, str],
    data_dir: Path,
    prefix: str,
    n_hvg: int = 2000,
    min_umi: int = 500,
) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """Stream-load Sci-Plex3 or Sci-Plex4 triplet sparse UMI counts, log1p-normalize, and extract top HVGs."""
    cache_npz = data_dir / f"{prefix}_hvg{n_hvg}.npz"
    cache_meta = data_dir / f"{prefix}_meta.parquet"
    cache_genes = data_dir / f"{prefix}_genes.txt"
    if cache_npz.exists() and cache_meta.exists() and cache_genes.exists():
        z = np.load(cache_npz)
        return z["X"].astype(np.float32), pd.read_parquet(cache_meta), cache_genes.read_text().splitlines()

    sub = data_dir / prefix
    sub.mkdir(parents=True, exist_ok=True)
    f_genes = download_file(urls["genes"], sub / "genes.txt.gz", min_bytes=1000)
    f_cells = download_file(urls["cells"], sub / "cells.txt.gz", min_bytes=500)
    f_pdata = download_file(urls["pdata"], sub / "pdata.txt.gz", min_bytes=5000)
    f_mat = download_file(urls["matrix"], sub / "matrix.txt.gz", min_bytes=50_000)

    genes_df = _read_genes_df(f_genes)
    cells_df = pd.read_csv(f_cells, sep=r"\s+", header=None, names=["cell", "sample"])
    pdata_df = pd.read_csv(f_pdata, sep=r"\s+", quotechar='"', index_col=0, low_memory=False)

    n_genes_all = len(genes_df)
    n_cells_all = len(cells_df)

    # Low-memory C-engine chunked triplet load
    X_sp = _load_sparse_triplet_matrix(f_mat, n_cells_all=n_cells_all, n_genes_all=n_genes_all)

    # Align pdata to cell annotations order
    cell_names = cells_df["cell"].astype(str).tolist()
    if "cell" in pdata_df.columns:
        pdata_df["cell"] = pdata_df["cell"].astype(str)
        pdata_df = pdata_df.drop_duplicates("cell").set_index("cell", drop=False)
    pdata_aligned = pdata_df.reindex(cell_names).reset_index(drop=True)

    # Filter cells with valid pData & sufficient UMI counts
    lib_size = np.asarray(X_sp.sum(axis=1)).ravel()
    valid = (lib_size >= min_umi) & (~pdata_aligned["cell_type"].isna().to_numpy())
    X_sp = X_sp[valid]
    pdata_aligned = pdata_aligned.loc[valid].reset_index(drop=True)
    lib_size = lib_size[valid]

    # Library size normalize (10,000) + log1p in sparse representation
    scale = (1e4 / np.maximum(lib_size, 1.0)).astype(np.float32)
    X_norm = sp.diags(scale) @ X_sp
    del X_sp
    X_norm.data = np.log1p(X_norm.data)

    # Select top HVGs by exact sparse variance E[X^2] - E[X]^2
    mean_g = np.asarray(X_norm.mean(axis=0)).ravel()
    mean_sq = np.asarray(X_norm.power(2).mean(axis=0)).ravel()
    var_g = np.maximum(0.0, mean_sq - mean_g ** 2)
    hvg_idx = np.sort(np.argsort(var_g)[::-1][:n_hvg])

    X_sub_sp = X_norm[:, hvg_idx]
    del X_norm
    X_dense = X_sub_sp.toarray().astype(np.float32)
    del X_sub_sp
    gene_names = genes_df.iloc[hvg_idx]["gene_short_name"].astype(str).tolist()

    np.savez_compressed(cache_npz, X=X_dense)
    pdata_aligned.to_parquet(cache_meta, index=False)
    cache_genes.write_text("\n".join(gene_names))
    return X_dense, pdata_aligned, gene_names


def load_sciplex4_dual_drug(
    data_dir: Path,
    n_hvg: int = 2000,
    min_umi: int = 500,
) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """Load Sci-Plex4 HDACi dual-drug combination screen with hashTable condition decoding."""
    cache_npz = data_dir / f"sciplex4_hvg{n_hvg}.npz"
    cache_meta = data_dir / "sciplex4_meta.parquet"
    cache_genes = data_dir / "sciplex4_genes.txt"
    if cache_npz.exists() and cache_meta.exists() and cache_genes.exists():
        z = np.load(cache_npz)
        return z["X"].astype(np.float32), pd.read_parquet(cache_meta), cache_genes.read_text().splitlines()

    sub = data_dir / "sciplex4"
    sub.mkdir(parents=True, exist_ok=True)
    f_genes = download_file(SCIPLEX4_URLS["genes"], sub / "genes.txt.gz", min_bytes=1000)
    f_cells = download_file(SCIPLEX4_URLS["cells"], sub / "cells.txt.gz", min_bytes=500)
    f_mat = download_file(SCIPLEX4_URLS["matrix"], sub / "matrix.txt.gz", min_bytes=50_000)
    f_hash = download_file(
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4150nnn/GSM4150379/suppl/GSM4150379_sciPlex4_A549_MCF7_HDACi_hashTable.out.txt.gz",
        sub / "hashTable.txt.gz",
        min_bytes=10_000,
    )

    genes_df = _read_genes_df(f_genes)
    cells_df = pd.read_csv(f_cells, sep=r"\s+", header=None, names=["cell", "sample"])
    ht = pd.read_csv(f_hash, sep=r"\s+", header=None, names=["sample", "cell", "oligo", "axis", "count"])
    ht_top = ht.sort_values("count", ascending=False).drop_duplicates("cell")

    X_sp = _load_sparse_triplet_matrix(f_mat, n_cells_all=len(cells_df), n_genes_all=len(genes_df))

    meta = cells_df.merge(ht_top[["cell", "oligo", "count"]], on="cell", how="left")
    # Parse oligo format: dose1_drug1_dose2_drug2_cellLine_plate_well
    d1_list, n1_list, d2_list, n2_list, cl_list = [], [], [], [], []
    for ol in meta["oligo"].fillna("").tolist():
        parts = str(ol).split("_")
        if len(parts) >= 5:
            try:
                d1, n1, d2, n2, cl = float(parts[0]), parts[1], float(parts[2]), parts[3], parts[4]
            except Exception:
                d1, n1, d2, n2, cl = 0.0, "UNK", 0.0, "UNK", "UNK"
        else:
            d1, n1, d2, n2, cl = 0.0, "UNK", 0.0, "UNK", "UNK"
        d1_list.append(d1)
        n1_list.append(n1)
        d2_list.append(d2)
        n2_list.append(n2)
        cl_list.append(cl)
    meta["dose1"] = d1_list
    meta["drug1"] = n1_list
    meta["dose2"] = d2_list
    meta["drug2"] = n2_list
    meta["cell_type"] = cl_list

    lib_size = np.asarray(X_sp.sum(axis=1)).ravel()
    valid = (lib_size >= min_umi) & meta["cell_type"].isin(["A549", "MCF7"]).to_numpy()
    X_sp = X_sp[valid]
    meta = meta.loc[valid].reset_index(drop=True)
    lib_size = lib_size[valid]

    scale = (1e4 / np.maximum(lib_size, 1.0)).astype(np.float32)
    X_norm = sp.diags(scale) @ X_sp
    X_norm.data = np.log1p(X_norm.data)

    mean_g = np.asarray(X_norm.mean(axis=0)).ravel()
    mean_sq = np.asarray(X_norm.power(2).mean(axis=0)).ravel()
    var_g = np.maximum(0.0, mean_sq - mean_g ** 2)
    hvg_idx = np.sort(np.argsort(var_g)[::-1][:n_hvg])
    X_dense = X_norm[:, hvg_idx].toarray().astype(np.float32)
    gene_names = genes_df.iloc[hvg_idx]["gene_short_name"].astype(str).tolist()

    np.savez_compressed(cache_npz, X=X_dense)
    meta.to_parquet(cache_meta, index=False)
    cache_genes.write_text("\n".join(gene_names))
    return X_dense, meta, gene_names


def prepare_perturb_splits(
    adata: ad.AnnData,
    seed: int = 42,
    n_hvg: int = 1000,
) -> Dict[str, object]:
    """Build standard Perturb-seq gene-perturbation matrix, co-expression priors, and combo_seen splits."""
    var_names = adata.var_names.astype(str).tolist()
    if "gene_name" in adata.var.columns:
        gene_symbols = adata.var["gene_name"].astype(str).tolist()
    else:
        gene_symbols = var_names

    cond_col = "condition" if "condition" in adata.obs.columns else "perturbation"
    conds = adata.obs[cond_col].astype(str).to_numpy()
    ctrl_mask = conds == "ctrl"

    if sp.issparse(adata.X):
        X_sp = adata.X.tocsr()
        m1 = np.asarray(X_sp.mean(axis=0)).ravel()
        m2 = np.asarray(X_sp.power(2).mean(axis=0)).ravel()
        var_g = np.maximum(0.0, m2 - m1 ** 2)
        top_idx = np.sort(np.argsort(var_g)[::-1][: min(n_hvg, X_sp.shape[1])])
        X_sub = X_sp[:, top_idx].toarray().astype(np.float32)
        ctrl_idx = np.where(ctrl_mask)[0][: min(int(ctrl_mask.sum()), 4000)]
        X_ctrl_full = X_sp[ctrl_idx].toarray().astype(np.float32)
    else:
        X_arr = np.asarray(adata.X, dtype=np.float32)
        var_g = np.var(X_arr, axis=0)
        top_idx = np.sort(np.argsort(var_g)[::-1][: min(n_hvg, X_arr.shape[1])])
        X_sub = X_arr[:, top_idx]
        X_ctrl_full = X_arr[ctrl_mask][: min(int(ctrl_mask.sum()), 4000)]
    genes_sub = [gene_symbols[i] for i in top_idx]

    # Extract all single gene names mentioned in conditions
    single_genes = set()
    parsed_conds: List[Tuple[str, ...]] = []
    for c in conds:
        if c == "ctrl":
            parsed_conds.append(())
            continue
        parts = [p for p in c.split("+") if p != "ctrl" and p != ""]
        parsed_conds.append(tuple(sorted(parts)))
        for p in parts:
            single_genes.add(p)
    pert_list = sorted(single_genes)
    pert_to_idx = {g: i for i, g in enumerate(pert_list)}

    U_all = np.zeros((len(conds), len(pert_list)), dtype=np.float32)
    for row_i, parts in enumerate(parsed_conds):
        for p in parts:
            U_all[row_i, pert_to_idx[p]] = 1.0

    # Gene co-expression features for every perturbed gene from control cells (zero test leakage!)
    ctrl_mean_full = X_ctrl_full.mean(axis=0)
    r_feat = min(48, X_sub.shape[1], max(2, len(X_ctrl_full) - 1))
    full_ctrl_c = X_ctrl_full - ctrl_mean_full
    # Gene-to-latent coexpression correlation with top PCA factors
    scores = full_ctrl_c @ (np.linalg.svd(full_ctrl_c, full_matrices=False)[2][:r_feat].T)  # (n_c, r_feat)
    scores_std = np.std(scores, axis=0, keepdims=True) + 1e-8
    scores_n = (scores - scores.mean(axis=0, keepdims=True)) / scores_std
    g_std = np.std(full_ctrl_c, axis=0, keepdims=True) + 1e-8
    g_n = (full_ctrl_c - full_ctrl_c.mean(axis=0, keepdims=True)) / g_std
    coexpr_all = (g_n.T @ scores_n) / max(1, len(full_ctrl_c) - 1)  # (G_all, r_feat)
    sym_to_full = {s: idx for idx, s in enumerate(gene_symbols)}
    pert_feats = np.zeros((len(pert_list), r_feat), dtype=np.float32)
    for pi, gname in enumerate(pert_list):
        if gname in sym_to_full:
            pert_feats[pi] = coexpr_all[sym_to_full[gname]]

    # Build GEARS-style unseen splits on unique conditions
    rng = np.random.default_rng(seed)
    unique_combos = sorted({c for c in set(conds) if c != "ctrl" and "+" in c and "ctrl" not in c.split("+")})
    unique_singles = sorted({c for c in set(conds) if c != "ctrl" and c not in unique_combos})
    all_pert_units = sorted(single_genes)
    rng.shuffle(all_pert_units)
    n_unseen_single = max(3, int(0.22 * len(all_pert_units)))
    unseen_units = set(all_pert_units[:n_unseen_single])
    seen_units = set(all_pert_units[n_unseen_single:])

    split_map: Dict[str, str] = {}
    for c in unique_combos:
        parts = [p for p in c.split("+") if p != "ctrl"]
        n_in_seen = sum(p in seen_units for p in parts)
        if n_in_seen == 2:
            split_map[c] = "combo_seen2"
        elif n_in_seen == 1:
            split_map[c] = "combo_seen1"
        else:
            split_map[c] = "combo_seen0"
    for c in unique_singles:
        parts = [p for p in c.split("+") if p != "ctrl"]
        if any(p in unseen_units for p in parts):
            split_map[c] = "unseen_single"
        else:
            split_map[c] = "train"

    # Keep ~55% of seen2 combos in train so training learns epistatic interaction!
    seen2_list = [c for c, sp_name in split_map.items() if sp_name == "combo_seen2"]
    rng.shuffle(seen2_list)
    for c in seen2_list[: max(1, int(0.55 * len(seen2_list)))]:
        split_map[c] = "train"

    return {
        "X": X_sub,
        "U": U_all,
        "conds": conds,
        "ctrl_mask": ctrl_mask,
        "pert_list": pert_list,
        "pert_to_idx": pert_to_idx,
        "pert_feats": pert_feats,
        "genes": genes_sub,
        "split_map": split_map,
    }
