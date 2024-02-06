import torch
import tqdm
import torch.nn.utils
import einops
from torch.nn import functional as F

class Model(torch.nn.Module):
    def __init__(self,n_features, n_hidden, S, n_instances=10, 
active=True):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty((n_instances, n_features, 
n_hidden)))
        torch.nn.init.xavier_normal_(self.W)
        self.b_final = torch.nn.Parameter(torch.zeros((n_instances, 
n_features)))
        self.active = active
        
        self.feature_probability = S
        self.n_instances = n_instances
        self.n_features = n_features
    
        
    def forward(self, features):

        hidden = torch.einsum("...if,ifh->...ih", features, self.W)

        out = torch.einsum("...ih,ifh->...if", hidden, self.W)

        zout = out + self.b_final
        if self.active:
            out = F.relu(out)
        return out
    
    def generate_batch(self, n_batch):
        feat = torch.rand((n_batch, self.n_instances, self.n_features))
        batch = torch.where(
            torch.rand((n_batch, self.n_instances, self.n_features)) <= 
self.feature_probability,
            feat,
            torch.zeros(()),
        )
        return batch


def learning(model, x, importances, n, epoch=10000, lr=0.001, 
batch_size=1024
             ):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


    for epoch in tqdm.tqdm(range(epoch)):

    

        optimizer.zero_grad()

        batch = model.generate_batch(batch_size)

        
        out = model(batch)
        
        error = (importances*(batch.abs() - out)**2)
        loss = einops.reduce(error, 'b i f -> i', 'mean').sum()
        
        loss.backward() 

        optimizer.step()
    return model, error

        
