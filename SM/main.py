
from dataset import *
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torchnet.dataset import TensorDataset
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
    root_data_path = ''
    mrna_data_path = root_data_path + 'filtered_mrnas.csv'
    mirna_data_path = root_data_path + 'five_per_mirnas.csv'

    ######### Loading Dataset ############
    print('loading data.......')
    mrna_data = RNA_Dataset(mrna_data_path, 'mrna')
    mirna_data = RNA_Dataset(mirna_data_path, 'mirna')

    ######### Train-Test Split ############
    '''num_cases = mrna_data.data.shape[0]
    t_size = np.round(num_cases*0.8).astype('int') #calculating training size
    #rest = num_cases-t_size
    #v_size = np.round(rest*0.5).astype(int) #calculating validation size
    t_id = np.random.choice(a=num_cases, size=t_size, replace=False) #training indices
    #s_id = np.delete(range(num_cases),t_id) 
    #v_id = np.random.choice(a=s_id, size=v_size, replace=False) #validation indices
    #tst_id = s_id[~np.isin(s_id,v_id)] #test indices
    tst_id = np.delete(range(num_cases),t_id) #test indices

    np.savetxt('train_indices.txt', t_id)
    #np.savetxt('validation_indices.txt', v_id)
    np.savetxt('test_indices.txt', tst_id)'''

    
    t_id = np.loadtxt("train_indices.txt").astype(int)
    tst_id = np.loadtxt("test_indices.txt").astype(int)
    
    print('splitting datasets.......')
    train_dataset = [Subset(mrna_data, t_id), Subset(mirna_data, t_id)]
    test_dataset = [Subset(mrna_data, tst_id), Subset(mirna_data, tst_id)]

    print('making dataloaders.......')
    train_loader = getDataLoaders(train_dataset, 64, True, True)
    test_loader = getDataLoaders(test_dataset, len(test_dataset[0]), False, False)

    mirna_features_no = mirna_data.shape[1]
    mrna_features_no = mrna_data.shape[1]


    for i in range(mirna_features_no):
        
        print('Mirna No: ', i + 1)
        input_params = (mrna_features_no, mirna_features_no)
        lr = 0.01
        epochs = 1
        hyper_params = (lr, epochs)
        mirna_no = i
        print('training model......')
        model  = train(SplicedAutoEncoder, train_loader, test_loader, input_params, hyper_params, mirna_no)

        print('Testing Model.......')
        test_pred = test(model, test_loader)
        test_pred_df = pd.DataFrame(list(test_pred.squeeze()))
        if i == 0:
            final_pred = test_pred_df
        if i > 0 :
            final_pred = pd.concat([final_pred,test_pred_df], axis=1)

    final_pred.to_csv('final_pred_5.csv', index= False)

    for i, batch_features in enumerate(test_loader):
        original = batch_features[1][1].detach().numpy()
        
    final_ori = pd.DataFrame(original)
    final_ori.to_csv('final_ori_5.csv', index=False)





