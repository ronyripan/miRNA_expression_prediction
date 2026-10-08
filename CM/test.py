
from scipy import stats


def test(model, test_loader):
    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    model.eval()
    
    for i, batch_features in enumerate(test_loader):
        data_lst = [batch_features[0][0], batch_features[1][0]]
        #target_lst = [batch_features[0][1], batch_features[1][1]]
        decoded = model(data_lst)
        #test_loss, pred_probas = loss(decoded[1][0], decoded[1][1], batch_features[1][1])
        pred = decoded.detach().numpy()
        #theta = decoded[1][1].detach().numpy()
        #original = batch_features[1][1].detach().numpy()
        #test_loss = test_loss.detach().numpy()
        #pred_probas = pred_probas.detach().numpy()
    #test_accuracy = stats.spearmanr(pred,original, axis = 0)  ### spearman correlation

    #return pred, theta, original, test_loss, pred_probas

    #return test_accuracy
    
    return pred