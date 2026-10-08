import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, num_inputs: int, num_units=16, activation=nn.PReLU):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_units = num_units

        self.encode1 = nn.Linear(self.num_inputs, 128)
        nn.init.xavier_uniform_(self.encode1.weight)
        self.bn1 = nn.BatchNorm1d(128)
        self.act1 = activation()

        self.encode2 = nn.Linear(128,64)
        nn.init.xavier_uniform_(self.encode2.weight)
        self.bn2 = nn.BatchNorm1d(64)
        self.act2 = activation()

        self.encode3 = nn.Linear(64, self.num_units)
        nn.init.xavier_uniform_(self.encode3.weight)
        self.bn3 = nn.BatchNorm1d(num_units)
        self.act3 = activation()

    def forward(self, x):
        x = self.act1(self.bn1(self.encode1(x)))
        x = self.act2(self.bn2(self.encode2(x)))
        x = self.act3(self.bn3(self.encode3(x)))
        return x
    
class Decoder(nn.Module):
    def __init__(
        self,
        num_outputs: 1,
        num_units: int = 16,
        intermediate_dim: int = 64,
        activation=nn.PReLU
    ):
        super().__init__()
        self.num_outputs = num_outputs
        self.num_units = num_units

        self.decode1 = nn.Linear(self.num_units, intermediate_dim)
        nn.init.xavier_uniform_(self.decode1.weight)
        self.bn1 = nn.BatchNorm1d(intermediate_dim)
        self.act1 = activation()

        self.decode2 = nn.Linear(intermediate_dim, 128)
        nn.init.xavier_uniform_(self.decode2.weight)
        self.bn2 = nn.BatchNorm1d(128)
        self.act2 = activation()

        self.decode3 = nn.Linear(128, self.num_outputs)
        nn.init.xavier_uniform_(self.decode3.weight)
        self.act3 = nn.ReLU()


    def forward(self, x):
        x = self.act1(self.bn1(self.decode1(x))) #64 x 64
        x = self.act2(self.bn2(self.decode2(x)))
        x = self.decode3(x) 
        return x

    
    
class SplicedAutoEncoder(nn.Module):
    """
    Spliced Autoencoder - where we have 4 parts (2 encoders, 2 decoders) that are all combined
    This does not work when you have chromsome split features
    """

    def __init__(
        self,
        input_dim1: int,
        input_dim2: int,
        hidden_dim: int = 16,
        seed=182822,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2

        self.encoder = Encoder(num_inputs=input_dim1, num_units=hidden_dim) #mrna encoder
        self.decoder = Decoder(num_outputs=1,num_units=hidden_dim) #mirna decoder

    def forward(self, x):
        assert isinstance(x, (tuple, list))
        assert len(x) == 2, "There should be two inputs to spliced autoencoder" #mrna and mirna

        encoded = self.encoder(x[0]) #->mrna embedding # 64x16

        decoded = self.decoder(encoded) #mrna->mirna

        return decoded
        

