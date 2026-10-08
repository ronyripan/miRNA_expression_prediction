from ast import mod
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
    sc.pp.normalize_total(adata, inplace=True)  ## size-normalize
    sc.pp.log1p(adata,copy=False) ##log-normalize
    sc.pp.scale(adata, zero_center=True, copy=False) #unit-variance-scaling

    #Babels's 0.5 clipping
    clip = 0.5
    if clip > 0:
        assert isinstance(clip, float) and 0.0 < clip < 50.0
        
        clip_low, clip_high = np.percentile(
                adata.X.flatten(), [clip, 100.0 - clip]
            )
        if clip_low == clip_high == 0:
            print("Skipping clipping, as clipping intervals are 0")
        else:
            assert (
                    clip_low < clip_high
                ), f"Got discordant values for clipping ends: {clip_low} {clip_high}"
            adata.X = np.clip(adata.X, clip_low, clip_high)
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