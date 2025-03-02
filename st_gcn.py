import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import data_utils 

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import random
import gc

def set_seed(seed=42):
    # Python random seed
    random.seed(seed)
    
    # NumPy random seed
    np.random.seed(seed)
    
    # PyTorch random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-gpu usage
    
    # Settings for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Random seed for the data-loader
    torch.Generator().manual_seed(seed)



# Define Spatial Graph Convolutional Layer
class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, adj_matrix):
        super().__init__()
        self.adj = adj_matrix
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        # x shape: (batch, time, nodes, channels)
        batch, time, nodes, _ = x.shape
        x = x.permute(0, 3, 1, 2)  # [batch, ch, time, nodes]
        
        # Spatial aggregation 
        x = torch.einsum('bctn,nm->bctm', (x, self.adj))  # Adjacency Matrix Aggregation
        x = self.conv(x)  # Tunnel Transformation
        return x.permute(0, 2, 3, 1).relu()  # Restore dimension and activation

# Define Temporal Convolutional Layer
class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                            kernel_size=(kernel_size, 1),
                            padding=(kernel_size//2, 0))
        
    def forward(self, x):
        # x shape: (batch, time, nodes, channels)
        x = x.permute(0, 3, 1, 2)  # [batch, ch, time, nodes]
        x = self.conv(x)
        return x.permute(0, 2, 3, 1).relu()

# Define ST-GCN Module
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, adj_matrix, temporal_kernel=3):
        super().__init__()
        self.spatial = SpatialGraphConv(in_channels, out_channels, adj_matrix)
        self.temporal = TemporalConv(out_channels, out_channels, temporal_kernel)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        
    def forward(self, x):
        res = x.permute(0, 3, 1, 2)
        if self.residual is not None:
            res = self.residual(res).permute(0, 2, 3, 1)
        else:
            res = x
        x = self.spatial(x)
        x = self.temporal(x)
        return x + res

# Define Network Structure
class ActionRecognitionNet(nn.Module):
    def __init__(self, adj_matrix, num_classes=20):
        super().__init__()
        self.block1 = STGCNBlock(3, 64, adj_matrix)
        self.block2 = STGCNBlock(64, 128, adj_matrix)
        self.block3 = STGCNBlock(128, 256, adj_matrix)
        
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Input shape: (batch, time, nodes, 3)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # 时空全局平均池化
        x = x.mean(dim=[1, 2])  # [batch, channels]
        return self.fc(x)

# Define adject matrix
def create_adjacency_matrix():
    # 20 nodes to construct skeleton
    adj = torch.eye(20)
    for i in range(19):
        adj[i, i+1] = 1
        adj[i+1, i] = 1
    return adj

# Training
def train_model(lr):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    adj = create_adjacency_matrix().to(device)
    
    #Load data
    train_set, test_set = data_utils.load_data()
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256)
    
    # Initialize the model
    model = ActionRecognitionNet(adj).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Training loop
    #f = open('st_gcn.txt','w')
    for epoch in range(300):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # release the variables
            del inputs, labels, outputs
         
        gc.collect()
        torch.cuda.empty_cache()
        
        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                correct += (outputs.argmax(1) == labels).sum().item()
        acc = correct / len(test_set) * 100
        
        #f.write('{:.3f} {:.3f}\n'.format(acc,loss.item()))
        
        print(f"Epoch {epoch+1}: Test Acc {acc:.3f}%  loss {loss.item():.3f}")
    
    #f.close()
    
    return acc

    

if __name__ == "__main__":
    set_seed(42)
    
    lr_lst = [0.0001,0.0005,0.001,0.002]
    
    #lr_lst = [0.001]
    
    results = []
    for lr in lr_lst:
        acc = train_model(lr)
        results.append(acc)
    
    print(results)
