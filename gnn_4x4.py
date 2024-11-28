import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv, global_mean_pool
import networkx as nx
import seaborn as sns

class GNN4x4(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32):
        super(GNN4x4, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, dropout=0.2)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.elu(x)

        self.attention_weights = self.conv1.att_weights

        x = global_mean_pool(x, data.batch)

        x = self.fc(x)
        return F.log_softmax(x, dim=1)

def matrix_to_graph(matrix):
    num_nodes = 16
    x = matrix.reshape(-1, 1).astype(np.float32)
    
    edge_index = []
    for i in range(4):
        for j in range(4):
            current_node = i * 4 + j
            
            neighbors = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < 4 and 0 <= nj < 4:
                        neighbor_node = ni * 4 + nj
                        neighbors.append(neighbor_node)
            
            for neighbor in neighbors:
                edge_index.append([current_node, neighbor])
                edge_index.append([neighbor, current_node])
    
    edge_index = np.array(edge_index).T
    
    return torch.FloatTensor(x), torch.LongTensor(edge_index)

def generate_data(num_samples):
    matrices = []
    labels = []
    
    for _ in range(num_samples):
        matrix = np.random.rand(4, 4)
        label = 1 if matrix.sum() > 8 else 0
        matrices.append(matrix)
        labels.append(label)
    
    return matrices, labels

def visualize_graph(matrix, edge_index, attention_weights=None):
    G = nx.Graph()
    
    for i in range(16):
        row, col = i // 4, i % 4
        G.add_node(i, value=matrix[row, col], pos=(col, -row))
    
    edge_list = edge_index.T.tolist()
    G.add_edges_from(edge_list)
    
    plt.figure(figsize=(12, 8))
    pos = nx.get_node_attributes(G, 'pos')
    
    if attention_weights is not None:
        edge_colors = attention_weights.detach().cpu().numpy()
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, 
                             edge_cmap=plt.cm.viridis, width=2)
    else:
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=1)
    
    node_colors = [G.nodes[i]['value'] for i in G.nodes()]
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                 node_size=500, cmap=plt.cm.viridis)
    
    plt.colorbar(nodes)
    
    labels = {i: f'{matrix[i//4, i%4]:.2f}' for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels)
    
    plt.title('4x4 Matrix as Graph with Attention Weights')
    plt.axis('equal')
    plt.show()

def train_gnn():
    torch.manual_seed(42)
    np.random.seed(42)

    num_train_samples = 100
    matrices, labels = generate_data(num_train_samples)

    train_data = []
    for i in range(num_train_samples):
        x, edge_index = matrix_to_graph(matrices[i])
        data = Data(x=x, edge_index=edge_index, 
                   y=torch.LongTensor([labels[i]]))
        train_data.append(data)

    model = GNN4x4()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    num_epochs = 50
    train_losses = []
    train_accs = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        
        for data in train_data:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.max(1)[1]
            correct += pred.eq(data.y).sum().item()

        avg_loss = total_loss / len(train_data)
        accuracy = correct / len(train_data)
        
        train_losses.append(avg_loss)
        train_accs.append(accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Loss: {avg_loss:.4f}')
            print(f'Accuracy: {accuracy:.4f}')

        if epoch % 10 == 0:
            sample_matrix = matrices[0]
            x, edge_index = matrix_to_graph(sample_matrix)
            visualize_graph(sample_matrix, edge_index, model.attention_weights)

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Training GNN on 4x4 matrices...")
    train_gnn()
