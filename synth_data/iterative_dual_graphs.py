import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pylab as plt
from scipy.sparse import csr_matrix
import seaborn as sns


N_A = 100
N_B = 100
T = 100
K = 5
num_neighbors = 3
A_df_path = 'A_graph.csv'
B_df_path = 'B_graph.csv'

A_neighbors = np.random.randint(0,N_B,size=(N_A,num_neighbors))
B_neighbors = np.random.randint(0,N_A,size=(N_B,num_neighbors))

def get_random_graph(n, p=0.5, directed=False):
    adj = np.random.rand(n, n)
    adj[adj < 1 - p] = 0
    adj[adj >= 1 - p] = 1

    if not directed:
        i = np.array([list(range(n))] * n)
        j = np.array([list(range(n))] * n).T
        relvant_index = i < j
        adj[relvant_index] = 0
        return nx.from_numpy_matrix(adj, create_using=nx.Graph())
    else:
        return nx.from_numpy_matrix(adj, create_using=nx.DiGraph())


# return nx.powerlaw_cluster_graph(n,int(n/2),p)

def next_graph(last_self_graphs, last_other_graphs, self_neighbors_in_other):
    # weighting time steps
    skew = 0.6
    n = len(last_self_graphs)
    t = np.linspace(0, n, n + 1)
    theta = (1 - skew) ** (n - t)
    theta /= sum(theta)

    last_self_graph = csr_matrix(nx.adjacency_matrix(last_self_graphs[0]).shape)
    for i, cur_graph in enumerate(last_self_graphs):
        last_self_graph += theta[i] * nx.adjacency_matrix(cur_graph)

    last_other_graph = csr_matrix(nx.adjacency_matrix(last_other_graphs[0]).shape)
    for i, cur_graph in enumerate(last_other_graphs):
        last_other_graph += theta[i] * nx.adjacency_matrix(cur_graph)

        # convolution on self
    last_self_graph = (np.sum(last_self_graph, axis=1) * np.sum(last_self_graph, axis=0)) / (
    last_self_graph.shape[0] * last_self_graph.shape[1])

    p = 1
    # adding other
    other_kernel = np.ones_like(last_self_graph)
    for i, other_neighbors in enumerate(self_neighbors_in_other):
        other_kernel[i, :] *= np.sum(last_other_graph[A_neighbors[i]])
    for j, other_neighbors in enumerate(self_neighbors_in_other):
        other_kernel[:, j] *= np.sum(last_other_graph[A_neighbors[i]])
    other_kernel = other_kernel / (
    A_neighbors.shape[1] * A_neighbors.shape[1] * last_other_graph.shape[0] * last_other_graph.shape[1])
    next_self_adj = p * last_self_graph + (1 - p) * other_kernel

    #     #adding noise
    #     lmb = 0.5
    #     next_self_adj = (1 - lmb) * last_self_graph + lmb * nx.adjacency_matrix(get_random_graph(last_self_graph.shape[0]))

    # sample randomly
    next_self_adj = next_self_adj / np.max(next_self_adj)
    thresh = 0  # np.random.rand(next_self_adj.shape[0],next_self_adj.shape[1])

    i = np.array([list(range(next_self_adj.shape[0]))] * next_self_adj.shape[1])
    j = np.array([list(range(next_self_adj.shape[0]))] * next_self_adj.shape[1]).T
    next_self_adj[i < j] = 0

    final_next_self_adj = np.random.binomial(1, next_self_adj)

    #     final_next_self_adj = np.zeros_like(next_self_adj)
    #     final_next_self_adj[next_self_adj-thresh>=np.percentile((next_self_adj-thresh)[i>=j],50)] = 1

    next_self_graph = nx.from_numpy_matrix(final_next_self_adj)

    return next_self_graph


def get_AB_df(t, A_nx, B_nx):
    A_df = pd.DataFrame(A_nx.edges(), columns=['from', 'to'])
    #     A_df['weight']=1
    #     A_df = A_df.groupby(['from','to']).agg({'weight':len}).reset_index()
    A_df['time'] = t

    B_df = pd.DataFrame(B_nx.edges(), columns=['from', 'to'])
    #     B_df['weight']=1
    #     B_df = B_df.groupby(['from','to']).agg({'weight':len}).reset_index()
    B_df['time'] = t

    return A_df, B_df

def get_dual_graphs():
    save = False
    verbose = True

    A_nxs, B_nxs = [], []
    for t in range(K):
        print(t, end='\r')
        A_nxs.append(get_random_graph(N_A))
        B_nxs.append(get_random_graph(N_B))

        if save:
            A_df, B_df = get_AB_df(t, A_nxs[-1], B_nxs[-1])
            A_df.to_csv(A_df_path, index=False, header=False, mode='a')
            B_df.to_csv(B_df_path, index=False, header=False, mode='a')

    for t in range(K, T):  # T-K):
        print(t, end='\r')
        if verbose:
            plt.figure()
            sns.heatmap(nx.adjacency_matrix(A_nxs[-1]).toarray())
            plt.show()

        A_nx, B_nx = next_graph(A_nxs[-K:], B_nxs[-K:], A_neighbors), \
                     next_graph(B_nxs[-K:], A_nxs[-K:], B_neighbors)

        if save:
            A_df, B_df = get_AB_df(t, A_nx, B_nx)
            A_df.to_csv(A_df_path, index=False, header=False, mode='a')
            B_df.to_csv(B_df_path, index=False, header=False, mode='a')

        A_nxs = A_nxs[1:] + [A_nx]
        B_nxs = B_nxs[1:] + [B_nx]


if if __name__ == '__main__':
    get_dual_graphs()