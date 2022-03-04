from torch_geometric.nn import Node2Vec
import os.path as osp
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch_geometric.datasets import Planetoid, KarateClub
from tqdm.notebook import tqdm
from src.arguments import args
import networkx as nx
import numpy as np
import pickle
from collections import namedtuple
from itertools import compress

torch.manual_seed(0)


def MFG():
    edge_index = torch.tensor(np.load(f"data/numpy/edge_inds.npy").T)
    y = torch.tensor(np.load(f"data/numpy/node_labels.npy"))
    num_nodes = len(y)
    num_classes = int(torch.max(y)) + 1
    train_mask, test_mask = assign_masks(num_nodes)
    Data = namedtuple('Data', ['edge_index', 'num_nodes', 'y', 'train_mask', 'test_mask', 'num_classes'])
    data = Data(edge_index, num_nodes, y, train_mask, test_mask, num_classes)
    return data


def assign_masks(num_nodes):
    train_ratio = 0.9
    idx = torch.randperm(num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[idx[:int(train_ratio*num_nodes)]] = True
    test_mask = torch.logical_not(train_mask)
    return train_mask, test_mask


def training_graph():
    data = MFG()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Node2Vec(data.edge_index, embedding_dim=64, walk_length=20,
                     context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(device)

    loader = model.loader(batch_size=1024, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=5*1e-3)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def test():
        model.eval()
        z = model()
        acc = model.test(z[data.train_mask], data.y[data.train_mask],
                         z[data.test_mask], data.y[data.test_mask],
                         max_iter=150)
        return acc

    for epoch in range(101):
        acc = test()
        loss = train()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')

    save_embedding = True
    if save_embedding:
        with torch.no_grad():
            model.eval()
            z = model(torch.arange(data.num_nodes, device=device))
            z = z.cpu().numpy()    
            np.save(f"data/numpy/embedding.npy", z)   


def plot_points():
    data = MFG()
    colors = ['blue', 'orange', 'red', 'black']

    with open('data/pickle/entity_names.pickle', 'rb') as handle:
        entities_list = pickle.load(handle)

    with open('data/pickle/entity_hashmap.pickle', 'rb') as handle:
        entity_hashmap = pickle.load(handle)

    selected_names = ['Metal', 'Polymer', 'Iron', 'Nickel', 'Selective_laser_sintering']
    selected_inds = [entity_hashmap[name] for name in selected_names]

    cache = True
    if cache:
        z = np.load(f"data/numpy/embedding_tsne.npy")
    else:
        z = np.load(f"data/numpy/embedding.npy")
        print(f"z.shape = {z.shape}")
        # z = TSNE(n_components=2, learning_rate=200).fit_transform(z)
        z = TSNE(n_components=2).fit_transform(z)
        np.save(f"data/numpy/embedding_tsne.npy", z)

    y = data.y.numpy()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    for i in range(data.num_classes - 1):
        line, = ax.plot(z[y == i, 0], z[y == i, 1], 'o', color=colors[i], picker=5)
        line.names = list(compress(entities_list, y == i))

    selected_z = np.take(z, selected_inds, axis=0)
    line, = ax.plot(selected_z[:, 0], selected_z[:, 1], 'o', color='purple', picker=5)
    line.names = selected_names

    plt.axis('off')
 
    def on_pick(event):
        thisline = event.artist
        xdata, ydata = thisline.get_data()
        ind = event.ind
        clicked_names = [thisline.names[i] for i in ind]
        print(f'on pick line: {xdata[ind]}, {ydata[ind]}')
        print(f"clicked_names: {clicked_names}")
 
    cid = fig.canvas.mpl_connect('pick_event', on_pick)


if __name__ == "__main__":
    # training_graph()
    plot_points() 
    plt.show()
