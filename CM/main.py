
from dataset import *
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torchnet.dataset import TensorDataset
import torch.optim as optim
import numpy as np
from autoencoders import *
from train import *
from test import *
import pandas as pd

######## making dataloaders ##########
def getDataLoaders(datasets, batch_size, shuffle, drop_last):
    datasets = TensorDataset(datasets) 

    dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last) #Shuffle here
        
    return dataloader

if __name__ == '__main__':
    
    mrna_data_path = "final_mrna.csv"
    mirna_data_path = "final_mirna.csv"

    ######### Loading Dataset ############
    print('loading data.......\n')
    mrna_data = RNA_Dataset(mrna_data_path, 'mrna')
    mirna_data = RNA_Dataset(mirna_data_path, 'mirna')

    ######### Train-Test Split ############
    t_id = np.loadtxt("train_indices.txt").astype(int)
    tst_id = np.loadtxt("test_indices.txt").astype(int)
    
    print('splitting datasets.......\n')
    train_dataset = [Subset(mrna_data, t_id), Subset(mirna_data, t_id)]
    test_dataset = [Subset(mrna_data, tst_id), Subset(mirna_data, tst_id)]
    
    bs = 64
    optimizer = optim.Adam
    
    print('making dataloaders.......\n')
    train_loader = getDataLoaders(train_dataset, bs, True, True)
    test_loader = getDataLoaders(test_dataset, len(test_dataset[0]), False, False)

    mirna_features_no = mirna_data.shape[1]
    mrna_features_no = mrna_data.shape[1]
    
    for i in range(mirna_features_no):
      print(f"miRNA no: {i+1}")
      input_params = (mrna_features_no, mirna_features_no)
      lr = 0.01
      epochs = 1
      mirna_no = i
      hyper_params = (lr, epochs, bs, optimizer)
  
      print('training model......')
      model  = train(SplicedAutoEncoder, train_loader, test_loader, input_params, hyper_params, mirna_no)
  
      print('Testing Model.......')
      test_pred = test(model, test_loader)
      
      test_pred_df = pd.DataFrame(list(test_pred.squeeze()))
      
      if i == 0:
        final_pred = test_pred_df
      if i > 0:
        final_pred = pd.concat([final_pred, test_pred_df], axis = 1)
      

        
    final_pred = final_pred.values
    print(final_pred.shape)
    
    for i, batch_features in enumerate(test_loader):
        original = batch_features[1][1].detach().numpy()
        
    print(original.shape)
    




