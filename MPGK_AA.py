import argparse
import networkx as nx
import numpy as np
import os
import sys
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
from nystrom import Nystrom
from utils import load_data


def create_tree(X, n_clusters, limit):
    T = nx.DiGraph()
    T.add_node(0)
    idx = [np.array(range(X.shape[0]))]
    ids = [0]
    max_id = 0
    idx2node = {}
    finished = False
    while not finished:
        finished = True
        idx_new = list()
        ids_new = list()
        for i in range(len(idx)):
            if idx[i].shape[0] >= n_clusters:
                kmeans = KMeans(n_clusters=n_clusters)
                kmeans.fit(X[idx[i]])
                kmeans.labels_
                unique = np.unique(kmeans.labels_)
                if unique.shape[0] > 1:
                    finished = False
                    for j in range(unique.shape[0]):
                        max_id += 1
                        T.add_node(max_id)
                        T.add_edge(ids[i], max_id)
                        ids_new.append(max_id)
                        if limit == None or nx.shortest_path_length(T, source=0, target=max_id) < limit:
                            idx_new.append(idx[i][np.where(kmeans.labels_==unique[j])])
                        else:
                            for j in range(idx[i].shape[0]):
                                idx2node[idx[i][j]] = max_id
                            
                else:
                    for j in range(idx[i].shape[0]):
                        idx2node[idx[i][j]] = ids[i]
            else:
                for j in range(idx[i].shape[0]):
                        idx2node[idx[i][j]] = ids[i]

        idx = idx_new
        ids = ids_new

    return T, idx2node


def compute_histograms(X, nbrs, n_clusters, limit):
    N = X.shape[0]

    T, idx2node = create_tree(X, n_clusters, limit)

    outdegs = T.out_degree()
    leaves = set([n for n, degree in outdegs if degree == 0])
    n_leaves = len(leaves)
    root = 0
    path = nx.shortest_path(T, source=root)

    depth = np.max(np.array([len(path[node]) for node in leaves]))
    for node in T.nodes():
        if node in leaves:
            T.node[node]['omega'] = 1
        else:
            T.node[node]['omega'] = len(path[node])/float(depth)

    n_features = T.number_of_nodes()
    hists = list()
    
    for i in range(N):
        hists.append(dict())
        for node in path[idx2node[i]]:
            if node in hists[i]:
                hists[i][node] += T.node[node]['omega']
            else:
                hists[i][node] = T.node[node]['omega']

    nbrs_hists = list()
    
    for i in range(N):
        nbrs_hists.append(dict())
        for neighbor in nbrs[i]:
            for node in path[idx2node[neighbor]]:
                if node in nbrs_hists[i]:
                    nbrs_hists[i][node] += T.node[node]['omega']
                else:
                    nbrs_hists[i][node] = T.node[node]['omega']

    return hists, nbrs_hists


def mpgk_aa(Gs, h, n_clusters, limit):
    N = len(Gs)
    if use_node_labels:
        d = Gs[0].node[list(Gs[0].nodes())[0]]['label'].size
    else:
        d = Gs[0].node[list(Gs[0].nodes())[0]]['attributes'].size

    idx = np.zeros(N+1, dtype=np.int64)
    nbrs = dict()
    ndata = []
    for i in range(N):
        n = Gs[i].number_of_nodes()
        idx[i+1] = idx[i] + n
        
        nodes = list(Gs[i].nodes())
        M = np.zeros((n,d))
        nodes2idx = dict()
        for j in range(idx[i], idx[i+1]):
            if use_node_labels:
                M[j-idx[i],:] = Gs[i].node[nodes[j-idx[i]]]['label']
            else:
                M[j-idx[i],:] = Gs[i].node[nodes[j-idx[i]]]['attributes']
            nodes2idx[nodes[j-idx[i]]] = j

        ndata.append(M)

        for node in nodes:
            nbrs[nodes2idx[node]] = list()
            for neighbor in Gs[i].neighbors(node):
                nbrs[nodes2idx[node]].append(nodes2idx[neighbor])

    graph_hists = list()
    X = np.vstack(ndata)

    for it in range(1,h+1):
        print("Iteration:", it)
        hists, nbrs_hists = compute_histograms(X, nbrs, n_clusters, limit)
        X = np.zeros((X.shape[0],200))

        ny = Nystrom(n_components=150)
        ny.fit(hists)
        X[:,:150] = ny.transform(hists)

        ny = Nystrom(n_components=50)
        ny.fit(nbrs_hists)
        X[:,150:] = ny.transform(nbrs_hists)
        
        graph_hists.append(list())
        for i in range(N):
            d = dict()
            for j in range(idx[i], idx[i+1]):
                for n in hists[j]:
                    if n in d:
                        d[n] += hists[j][n]
                    else:
                        d[n] = hists[j][n]
            graph_hists[it-1].append(d)
    
    K = np.zeros((N, N))
    for it in range(h):
        for i in range(N):
            for j in range(i,N):
                for n in graph_hists[it][i]:
                    if n in graph_hists[it][j]:
                        K[i,j] += min(graph_hists[it][i][n], graph_hists[it][j][n])
                K[j,i] = K[i,j]
    
    return K

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('ROOT', type=str, help='Root directory for data sets')
    parser.add_argument('NAME', type=str, help='Data set name')
    parser.add_argument(
        '-n',
        type=int,
        default=4,
        help='Maximum number of iterations'
    )

    parser.add_argument(
        '-l', '--labels',
        action='store_true',
        default=False,
        help='If set, uses node labels'
    )

    parser.add_argument(
        '-a', '--attributes',
        action='store_true',
        default=False,
        help='If set, uses node attributes'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output directory'
    )

    args = parser.parse_args()

    # read the parameters
    ds_name = args.NAME
    n_iter = int(args.n)
    use_node_labels = args.labels
    use_node_attributes = args.attributes

    # Notice that the function returns potentially a new value for the
    # node labels and node attributes in case the client requested 'em
    # but they cannot be found.
    graphs, labels, use_node_labels, use_node_attributes = load_data(
        ds_name,
        args.ROOT,
        use_node_labels,
        use_node_attributes
    )

    # Create kernel matrices for a different number of maximum
    # iterations
    matrices = {
            str(i): mpgk_aa(graphs, n_iter, 4, 8)
            for i in range(1, n_iter + 1)
    }

    # Don't forget to store the labels!
    matrices['y'] = labels

    os.makedirs(args.output, exist_ok=True)

    # TODO: make it possible to override this?
    np.savez(
        os.path.join(args.output, 'MP.npz'),
        **matrices
    )
