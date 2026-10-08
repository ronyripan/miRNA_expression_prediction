import torch
from torch.distributions import Normal
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision.utils import save_image
import pandas as pd
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

from preprocessing import *

def load_data(path, modality):
    
    raw_df = pd.read_csv(path, index_col=0)

    if modality == 'mirna':
      #raw_df = mirna_sampling(raw_df)
      normalized_array = preprocess_mirna(raw_df)
      count = csr_matrix(normalized_array).astype('float32')
      return count, count
    else:
      raw_df1 = pd.read_csv(path, index_col=0)
      normalized_array = preprocess_mrna(raw_df)
      size_normalized_array = size_normalize(raw_df1)
      count = csr_matrix(normalized_array).astype('float32')
      sn_count = csr_matrix(size_normalized_array).astype('float32')
      return count, sn_count



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

