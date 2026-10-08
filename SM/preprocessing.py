import pandas as pd
import numpy as np
from anndata import AnnData
import scanpy as sc
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def preprocess_mirna(df):
    adata = AnnData(df.values)
    sc.pp.log1p(adata,copy=False) ##log-normalize
    return adata.X

def preprocess_mrna(df):
    adata = AnnData(df.values)
    #sc.pp.normalize_total(adata, inplace=True)  ## size-normalize
    sc.pp.log1p(adata,copy=False) ##log-normalize
    sc.pp.scale(adata, zero_center=True, copy=False) #unit-variance-scaling
    return adata.X

def size_normalize(df):
    adata = AnnData(df.values)
    sc.pp.normalize_total(adata, inplace=True)  ## size-normalize
    return adata.X

def mirna_sampling(mirna_df):
    threshold = 0.5 * len(mirna_df)
    columns = mirna_df.columns[(mirna_df == 0).sum() > threshold] ###columns_with_more_than_50_percent_zeroes
    mirna_df.drop(columns, axis = 1, inplace=True)
    return mirna_df