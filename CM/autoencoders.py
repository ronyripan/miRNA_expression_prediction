import torch
import torch.nn as nn
from activations import *

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
    
class mirna_decoder(nn.Module):
    def __init__(
        self,
        num_outputs: int,
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

class mrna_decoder(nn.Module):
    def __init__(
        self,
        num_outputs: int,
        num_units: int = 32,
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
        self.act3 = Exp() #mean
        self.act4 = ClippedSoftplus() #dispersion


    def forward(self, x):
        x = self.act1(self.bn1(self.decode1(x))) #64 x 64
        x = self.act2(self.bn2(self.decode2(x)))
        x = self.decode3(x)  
        x_m = self.act3(x) #mean
        x_d = self.act4(x) #dispersion
        return x_m, x_d
    
    
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

        self.encoder1 = Encoder(num_inputs=input_dim1, num_units=hidden_dim) #mrna encoder
        self.encoder2 = Encoder(num_inputs=input_dim2, num_units=hidden_dim) #mirna encoder

        self.decoder1 = mrna_decoder(num_outputs=input_dim1,num_units=16) #mrna to mrna decoder
        self.decoder2 = mirna_decoder(num_outputs=1,num_units=16) #mirna to mirna decoder

    def forward(self, x):
    
        if self.training:
          assert isinstance(x, (tuple, list))
          assert len(x) == 2, "There should be two inputs to spliced autoencoder" #mrna and mirna
  
          encoded1 = self.encoder1(x[0]) #mrna embedding # 64x16
          encoded2 = self.encoder2(x[1]) #mirna embedding # 64 x 16
  
          decoded1 = self.decoder1(encoded1) #mrna->mrna
          decoded2 = self.decoder1(encoded2) #mirna->mrna
          decoded3 = self.decoder2(encoded2) #mirna->mirna
          decoded4 = self.decoder2(encoded1) #mrna->mirna ..... our focus
  
          retval1 = *decoded1, encoded1 
          retval2 = *decoded2, encoded2
          retval3 = decoded3, encoded2
          retval4 = decoded4, encoded1   
          
          return retval1, retval2, retval3, retval4
        else:
          encoded1 = self.encoder1(x[0]) #mrna embedding # 64x16
          decoded4 = self.decoder2(encoded1) #mrna->mirna ..... our focus
          return decoded4
        
        

