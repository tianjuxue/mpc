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
from collections import namedtuple

torch.manual_seed(0)


def Karate():
    manual = False
    if manual:
        # TODO: not complete, we need data.y
        G = nx.karate_club_graph()
        edges = G.edges()
        nodes = G.nodes()
        edge_index = torch.tensor(np.array(edges).T)
        Data = namedtuple('Data', ['edge_index', 'num_nodes'])
        data = Data(edge_index, len(nodes))
        dataset = [data]
    else:
        dataset = KarateClub()
    return dataset

 
def exp():
    dataset = Karate()
    data = dataset[0]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(data.edge_index, embedding_dim=8, walk_length=6,
                     context_size=3, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(device)

    loader = model.loader(batch_size=10, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.05)

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


    for epoch in range(30):
        loss = train()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')


    @torch.no_grad()
    def plot_points(colors):
        model.eval()
        z = model(torch.arange(data.num_nodes, device=device))

        z = TSNE(n_components=2, learning_rate=10).fit_transform(z.cpu().numpy())

        # pca = PCA(n_components=2)
        # z = pca.fit_transform(z)

        y = data.y.cpu().numpy()

        plt.figure(figsize=(8, 8))
        for i in range(dataset.num_classes):
            plt.scatter(z[y == i, 0], z[y == i, 1], s=50, color=colors[i])
        plt.axis('off')
        plt.show()

    colors = ['blue', 'red', 'orange', 'purple']
    plot_points(colors)


if __name__ == "__main__":
    exp()
