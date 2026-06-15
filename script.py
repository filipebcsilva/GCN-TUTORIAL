import os
import glob
import numpy as np
import scipy.io as sio
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv
import torch.optim as optim
from torch.utils.data import DataLoader
import itertools



def sharp_sigmoid(x, sharpness=50, threshold=0.1):
    return torch.sigmoid(sharpness * (x - threshold))

class GestureSequenceDataset(Dataset):
    def __init__(self, root_dir, transform=None, pre_transform=None):
        super(GestureSequenceDataset, self).__init__(root_dir, transform, pre_transform)
        self.root_dir = root_dir
        self.mat_files = glob.glob(os.path.join(root_dir, '*.mat'))
        
        self.label_to_id = {} 
        self.current_id = 0 
        
        self.data_list = self._process_all_files()

    def _process_all_files(self):
        all_sequences = []
        
        for filepath in self.mat_files:
            mat_data = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)
            video = mat_data['Video']
            frames = np.atleast_1d(video.Frames)
            
            if not hasattr(video, 'Labels'):
                continue
                
            labels = np.atleast_1d(video.Labels)
            
            for segment in labels:
                nome_gesto = segment.Name
                start_idx = segment.Begin - 1
                end_idx = segment.End 
                
                frames_do_gesto = frames[start_idx:end_idx]
                
                sequencia_coordenadas = []
                
                for frame in frames_do_gesto:
                    if hasattr(frame, 'Skeleton') and hasattr(frame.Skeleton, 'PixelPosition'):
                        pos = frame.Skeleton.PixelPosition
                        if isinstance(pos, np.ndarray) and pos.ndim == 2 and pos.shape[0] == 20:
                            sequencia_coordenadas.append(pos)
                
                if len(sequencia_coordenadas) > 0:
                    
                    x_seq = torch.tensor(np.array(sequencia_coordenadas), dtype=torch.float)
                    
                    centro_de_massa = x_seq.mean(dim=1, keepdim=True) 

                    x_seq = x_seq - centro_de_massa

                    x_seq = x_seq / (x_seq.std() + 1e-5)

                    if nome_gesto not in self.label_to_id:
                        self.label_to_id[nome_gesto] = self.current_id
                        self.current_id += 1
                        
                    y_val = self.label_to_id[nome_gesto]
                    y = torch.tensor([y_val], dtype=torch.long)
                    
                    empty_edge_index = torch.empty((2, 0), dtype=torch.long)
                    
                    graph_data = Data(x=x_seq, edge_index=empty_edge_index, y=y)
                    all_sequences.append(graph_data)
                        
        return all_sequences

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

def preencher_lote_collate_fn(lista_de_grafos):
    xs = [grafo.x for grafo in lista_de_grafos] 
    ys = [grafo.y for grafo in lista_de_grafos] 
    
    tamanhos_reais = torch.tensor([len(x) for x in xs], dtype=torch.long)
    
    x_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)
    
    y_batched = torch.cat(ys)
    
    return x_padded, y_batched, tamanhos_reais

class GestureGSLModel(nn.Module):
    def __init__(self, num_nodes=20, in_features=2, num_classes=20,hiddem_dim = 64):
        super().__init__()
        
        adj = torch.rand((num_nodes, num_nodes)) * 0.04 + 0.01
        self.adj_param = nn.Parameter(adj)
        
        self.gcn1 = DenseGCNConv(in_features, hiddem_dim)
        # self.gcn2 = DenseGCNConv(hiddem_dim, 2*hiddem_dim)
        self.classifier = nn.Linear(hiddem_dim , num_classes)

    def normalize_adj(self, adj):
        adj_sym = adj + adj.transpose(-2, -1)
        
        eye = torch.eye(adj_sym.size(-1), device=adj_sym.device)
        adj_hat = adj_sym + eye
        deg = adj_hat.sum(dim=-1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return deg_inv_sqrt.unsqueeze(-1) * adj_hat * deg_inv_sqrt.unsqueeze(1)
    
    def forward(self, x_padded, mask):
        B, T, N, C = x_padded.shape #Extract(Batch,Time,Nodes,Features)
        
        adj = sharp_sigmoid(self.adj_param,10,0.1)
        adj_norm = self.normalize_adj(adj) 
        
        adj_batch = adj_norm.unsqueeze(0).expand(B * T, N, N) #Copy the adj batch X time to pass trough the gcn
        
        x = x_padded.view(B * T, N, C) # Extract all the frames of the videos
        
        x = self.gcn1(x, adj_batch, add_loop=False)
        
        # x = F.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        
        # x = self.gcn2(x, adj_batch, add_loop=False)
        x = F.relu(x)
        
        x = x.view(B, T, N, -1) # Make x back to the normal size
        
        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).expand_as(x) # apply a mask to exclude padding frames
        x = x * mask_expanded.float() 
        
        x_spatial = x.mean(dim=2) # Pooling of the nodes
        
        soma_temporal = x_spatial.sum(dim=1)
        tamanhos_reais = mask.sum(dim=1, keepdim=True).float() 
        x_temporal = soma_temporal / tamanhos_reais.clamp(min=1) # Pooling of the frames 
        
        out = self.classifier(x_temporal) 
        
        return out

def train(model, loader, optimizer, criterion, device):
    model.train()
    loss_treino_acumulada = 0.0
    acertos_treino = 0
    total_treino = 0
    
    for x_padded, y_batched, tamanhos_reais in loader:
        x_padded = x_padded.to(device)
        y_batched = y_batched.to(device)
        tamanhos_reais = tamanhos_reais.to(device)
        
        B, T = x_padded.shape[0], x_padded.shape[1]
        posicoes = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        mask = posicoes < tamanhos_reais.unsqueeze(1)
        
        optimizer.zero_grad()
        
        previsoes = model(x_padded, mask)
        
        loss = criterion(previsoes, y_batched)  + sharp_sigmoid(model.adj_param, 30, 0.1).sum() * 0.0000025
        
        loss.backward()
        
        optimizer.step()
        
        loss_treino_acumulada += loss.item()
        
        _, classe_prevista = torch.max(previsoes, 1)
        acertos_treino += (classe_prevista == y_batched).sum().item()
        total_treino += y_batched.size(0)

    acc_treino = acertos_treino / total_treino
    loss_media_treino = loss_treino_acumulada / len(loader)
    
    return loss_media_treino, acc_treino

def evaluate(model, loader, criterion, device):
    
    model.eval()
    loss_val_acumulada = 0.0
    acertos_val = 0
    total_val = 0
    
    with torch.no_grad(): 
        for x_padded, y_batched, tamanhos_reais in loader:
            x_padded = x_padded.to(device)
            y_batched = y_batched.to(device)
            
            tamanhos_reais = tamanhos_reais.to(device)
            
            B, T = x_padded.shape[0], x_padded.shape[1]
            posicoes = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            mask = posicoes < tamanhos_reais.unsqueeze(1)
            
            previsoes = model(x_padded, mask)
            loss = criterion(previsoes, y_batched)
            
            loss_val_acumulada += loss.item()
            _, classe_prevista = torch.max(previsoes, 1)
            acertos_val += (classe_prevista == y_batched).sum().item()
            total_val += y_batched.size(0)
            
    acc_val = acertos_val / total_val
    loss_media_val = loss_val_acumulada / len(loader)
    
    return loss_media_val, acc_val
dataset = GestureSequenceDataset(root_dir='apenas_mats_train')


total_amostras = len(dataset)

tamanho_treino = int(0.70 * total_amostras)
tamanho_val = int(0.15 * total_amostras)
tamanho_teste = total_amostras - tamanho_treino - tamanho_val 

print(f"Divisão: Treino ({tamanho_treino}), Validação ({tamanho_val}), Teste ({tamanho_teste})")

dataset_treino, dataset_val, dataset_teste = random_split(
    dataset, 
    [tamanho_treino, tamanho_val, tamanho_teste],
)

hiddem_dims = [32,64, 128]
batch_sizes = [16, 32, 64]
learning_rates = [0.001,0.005,0.01,0.05] 
num_epochs = 1000

combinacoes = list(itertools.product(hiddem_dims, batch_sizes, learning_rates))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
melhor_acuracia = 0.0
melhores_parametros = {}
criterion = nn.CrossEntropyLoss()
        
print(f"Iniciando busca em grade com {len(combinacoes)} combinações...")
print("-" * 50)

for hiddem_dim, batch_size, lr in combinacoes:
    print(f"Testando: hiddem_dim={hiddem_dim} | batch_size={batch_size} | lr={lr}")
    
    loader_treino = DataLoader(
    dataset_treino, 
    batch_size=batch_size, 
    shuffle=True, 
    collate_fn=preencher_lote_collate_fn
    )

    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, collate_fn=preencher_lote_collate_fn)
    
    model = GestureGSLModel(
        num_nodes=20, 
        in_features=2, 
        num_classes=20, 
        hiddem_dim=hiddem_dim
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
       
    for epoch in range(num_epochs):
            
        loss_media_treino, acc_treino = train(model, loader_treino, optimizer, criterion, device)
        
        loss_media_val, acc_val = evaluate(model, loader_val, criterion, device)
        
        if epoch % 10 == 0 or epoch == 0:
            print(f"Época [{epoch+1}/{num_epochs}] | "
                    f"Loss Treino: {loss_media_treino:.4f} - Acc Treino: {acc_treino:.4f} | "
                    f"Loss Val: {loss_media_val:.4f} - Acc Val: {acc_val:.4f}")

        
        if acc_treino > melhor_acuracia:
            melhor_acuracia = acc_treino
            melhores_parametros = {
                'hiddem_dim': hiddem_dim,
                'batch_size': batch_size,
                'lr': lr
            }
    
    result = f"Acc Treino: {acc_treino:.4f}, Acc Val: {acc_val:.4f}, Hiddem Dim: {hiddem_dim}, Batch size: {batch_size}, Lr: {lr}"
    
    with open("log_2camada_sharp30.txt", "a") as arquivo:
        arquivo.write(result + "\n") 
    

print("BUSCA FINALIZADA!")
print(f"Melhor Acurácia: {melhor_acuracia:.4f}")
print("Melhores Parâmetros:", melhores_parametros)