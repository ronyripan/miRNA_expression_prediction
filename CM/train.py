from babel_loss import QuadLoss
#from contrastive_loss_pytorch import ContrastiveLoss
#from loss_functions import rmse_loss
import torch.optim as optim
import torch
import torch.nn as nn

def train(model, train_loader, validation_loader, input_params, hyper_params, mirna_no):
    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    mrna_features_no, mirna_features_no = input_params
    lr, epochs, bs, optimize = hyper_params
    n_epochs = epochs
    net = model(mrna_features_no, mirna_features_no)
    net.train()
    net_loss = QuadLoss()
    #con_loss = ContrastiveLoss(batch_size = bs) #contrastive loss
    mse_loss = nn.MSELoss()
    optimizer = optimize(net.parameters(), lr = lr)
    loss_list = []
    model_dict = []
    cur_min = 10000
    for epoch in range(n_epochs):
        loss = 0
        epoch_dict = {}
        for i, batch_features in enumerate(train_loader):
            #batch_features = batch_features.cpu()
            optimizer.zero_grad()
            #compute reconstructions
            data_lst = [batch_features[0][0], batch_features[1][0]]
            target_lst = [batch_features[0][1], batch_features[1][1]]
            decoded = net(data_lst) #automatically run the forward function
            
            preds1 , preds2, preds3, preds4 = decoded
            
            mirna_decoded1, emb1 = preds3
            mirna_decoded2, emb2 = preds4
    
            nb_loss = net_loss(decoded, target_lst) #return summation of two loss
            
            input_mirna_1 = mirna_decoded1.squeeze()
            input_mirna_2 = mirna_decoded2.squeeze()
            target_mirna = target_lst[1][:,mirna_no]
            rmse_loss = torch.sqrt(mse_loss(input_mirna_1, target_mirna)) + torch.sqrt(mse_loss(input_mirna_2, target_mirna))
            
            train_loss = nb_loss + rmse_loss 

            #compute accumulated gradients
            train_loss.backward()

            #perform parameter update based on current gradients

            optimizer.step()

            #add the mini_batch training loss to epoch loss

            loss += train_loss.item()

            #compute the epoch training loss

        loss = loss / len(train_loader)
        #loss_list.append(loss)

        '''with torch.no_grad():  
            for i, batch_features in enumerate(validation_loader):
                data_lst = [batch_features[0][0], batch_features[1][0]]
                target_lst = [batch_features[0][1], batch_features[1][1]]
                decoded = net(data_lst) #automatically run the forward function
                #compute training reconstruction loss
                validation_loss = net_loss(decoded, target_lst)

        if validation_loss <= cur_min:
            cur_min = validation_loss
            best_model = epoch

        
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': loss,
            'validation_loss': validation_loss
            }, '/lustre/fs1/home/rripan/mirna_prediction/CM_merged_embedding_stacked_sidewise/output/rev_2_splits/test_set_2/training_outputs/{}_model.pth'.format(epoch))'''

        #model_dict.append(epoch_dict)
        if epoch % 5 == 0:
            print("epoch : {}/{}, loss = {:.6f}".format(epoch+1, n_epochs, loss))

    print("epoch : {}/{}, loss = {:.6f}".format(epoch+1, n_epochs, loss))
    #print("Best Epoch: ", best_model)
    #torch.save(model_dict, 'model.pth')
    #print("Loss: {}".format(loss))
    return net