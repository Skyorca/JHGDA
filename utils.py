import torch
import networkx as nx
import torch
import numpy as np



def generate_core_view(path, kmax=1):
    """
    :return: k-core-views of original graph
    """
    g = nx.read_edgelist(path, nodetype=int, create_using=nx.Graph)
    mapping = {}
    r_mapping = {}
    nodes = list(g.nodes)
    for idx in range(len(nodes)):
        mapping[nodes[idx]] = idx
        r_mapping[idx] = nodes[idx]
    g = nx.relabel_nodes(g,mapping=mapping)
    g.remove_edges_from(nx.selfloop_edges(g))
    print(path, g.number_of_edges())
    views = []
    for i in range(1,kmax+1):
        core = nx.k_core(g,k=i)
        # NOTE : nx to scipy sparse matrix要求node ordering，比较麻烦，不如直接拿节点映射后的ID
        row = np.array([x[0] for x in core.edges])
        col = np.array([x[1] for x in core.edges])
        # NOTE 为什么必须要输入双向边才有好的效果, 而且与sparse matrix结果不一致
        indices = np.hstack([np.vstack([row, col]),np.vstack([col, row])])
        self_loops = np.vstack([np.array(core.nodes),np.array(core.nodes)])
        indices = np.hstack([indices, self_loops])
        edge_index = torch.tensor(indices,dtype=torch.long)
        views.append(edge_index)
    return g,views,r_mapping

def generate_feture(graph, max_degree):
    features = torch.zeros([graph.number_of_nodes(), max_degree])
    # nodeID是0开始的数组编号，因此可以这么写
    for i in range(graph.number_of_nodes()):
        try:
            # note degree比in_degree效果好太多了
            features[i][min(graph.degree[i], max_degree-1)] = 1
        except:
            features[i][0] = 1
    return features

def generate_label(path, r_mapping):
    labels = dict()
    with open(path) as IN:
        IN.readline()
        for line in IN:
            tmp = line.strip().split(' ')
            labels[int(tmp[0])] = int(tmp[1])
    y = []
    for idx, nodeid in r_mapping.items():
        y.append(labels[nodeid])
    y = torch.tensor(y)
    return y

def get_data(domain, kmax=1):
    graph, views, r_mapping = generate_core_view(f'data/{domain}-airports.edgelist',kmax=kmax)
    feature = generate_feture(graph, 8)
    label = generate_label(f"data/labels-{domain}-airports.txt", r_mapping)
    return feature, label, views

def to_onehot(label_matrix, num_classes, device):
    identity = torch.eye(num_classes).to(device)
    onehot = torch.index_select(identity, 0, label_matrix)
    return onehot


def entropy(x:torch.Tensor, reduction='mean'):
    """x: row-softmax probability matrix"""
    # try:
    # int(x.0000) \neq x 可能由于浮点运算导致的
    #     assert int(torch.sum(torch.sum(x,dim=1)).item())==x.size()[0]  # check row-wise softmax
    # except:
    #     print(x,torch.sum(torch.sum(x,dim=1)),x.size())
    #     raise AssertionError
    eps = 1e-7
    log_x = torch.log(x+eps)
    entropy_x = torch.sum(-x*log_x,dim=1)
    if reduction=='mean':
        return torch.mean(entropy_x)


def accuracy_citation(y_true, y_pred):
    top_k_list = torch.sum(y_true,dim=1, dtype=int)
    prediction = []
    for i in range(y_true.shape[0]):
        pred_i = torch.zeros(y_true.shape[1])
        pred_i[torch.argsort(y_pred[i,:])[-top_k_list[i]:]]=1
        prediction.append(pred_i.reshape(1,-1))
    prediction = torch.vstack(prediction)
    c = 0
    for i in range(y_true.shape[0]):
        if torch.sum(y_true[i,:]@prediction[i,:].T)>0: c+= 1
    return c/y_true.shape[0]