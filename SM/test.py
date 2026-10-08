
from scipy import stats


def test(model, test_loader):
    model.eval()
    
    for i, batch_features in enumerate(test_loader):
        data_lst = [batch_features[0][0], batch_features[1][0]]
        target_lst = [batch_features[0][1], batch_features[1][1]]
        decoded = model(data_lst)
        pred = decoded.detach().numpy()
        #original = batch_features[1][1].detach().numpy()
    
    return pred