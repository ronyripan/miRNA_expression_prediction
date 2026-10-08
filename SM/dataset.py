import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset
from preprocessing import *

def load_data(path, modality):
    
    raw_df = pd.read_csv(path, index_col=0)

    if modality == 'mirna':
      #raw_df = mirna_sampling(raw_df)
      normalized_array = preprocess_mirna(raw_df)
      count = csr_matrix(normalized_array).astype('float32')
      return count, count
    else:
      normalized_array = preprocess_mrna(raw_df)
      count = csr_matrix(normalized_array).astype('float32')
      return count, count



class RNA_Dataset(Dataset):

  def __init__(self, data_path, modality):
    self.data, self.sn_count = load_data(data_path, modality)
    self.indices = None
    self.n_cases, self.n_genes = self.data.shape
    self.shape = self.data.shape

  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, index):
    data = self.data[index]
    target = self.sn_count[index]
    if type(data) is not np.ndarray:
      data = data.toarray().squeeze()

    if type(target) is not np.ndarray:
      target = target.toarray().squeeze()
    return torch.tensor(data), torch.tensor(target)

  def info(self):
    print("\n===========================")
    print("Dataset Info")
    print('Case no: {}\nGene number: {}'.format(self.n_cases, self.n_genes))
    print('===========================\n')

