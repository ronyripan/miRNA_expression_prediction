import torch.optim as optim
import torch
import torch.nn as nn

def rmse_loss(decoded, original):
        mse = nn.MSELoss()
        mirna_original = original

        mrna_mirna = decoded #mirna
        
        loss = torch.sqrt(mse(mrna_mirna, mirna_original))

        return loss

def train(model, train_loader, validation_loader, input_params, hyper_params, mirna_no):
    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    mrna_features_no, mirna_features_no = input_params
    lr, epochs = hyper_params
    n_epochs = epochs
    net = model(mrna_features_no, mirna_features_no)
    net.train()
    #net_loss = rm()
    optimizer = optim.Adam(net.parameters(), lr = lr)
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
            #print(decoded)
            #compute training reconstruction loss
            train_loss = rmse_loss(decoded.squeeze(), target_lst[1][:,mirna_no])

            #compute accumulated gradients
            train_loss.backward()

            #perform parameter update based on current gradients

            optimizer.step()

            #add the mini_batch training loss to epoch loss

            loss += train_loss.item()

            #compute the epoch training loss

        loss = loss / len(train_loader)
        #loss_list.append(loss)
        #model_dict.append(epoch_dict)
        if epoch % 5 == 0:
            print("epoch : {}/{}, loss = {:.6f}".format(epoch+1, n_epochs, loss))

    print("epoch : {}/{}, loss = {:.6f}".format(epoch+1, n_epochs, loss))
    return net