"""
伪标签直接用naive的方式
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_sparse import SparseTensor, spmm
from utils import *
from pseudo_labeling import *


class GCNEncoder(nn.Module):
    """
    $Z^L=GNN_{emb}(X^L,A^L)$
    $GNN_{emb}$ has 2 conv layers
    This encoder is shared between domains
    """
    def __init__(self,param,drop_p,device):
        n_layer, input_dim, hidden_dims = param[0],param[1],param[2]
        super(GCNEncoder, self).__init__()
        self.layers = []
        self.conv1 = GCNConv(input_dim, hidden_dims[0]).to(device)
        self.layers.append(self.conv1)
        for i in range(n_layer-1):
            self.layers.append(GCNConv(hidden_dims[i], hidden_dims[i+1]).to(device))
        self.relu = nn.ReLU()
        self.drop_p = drop_p
    def forward(self, x, edge_index, edge_weight):
        feat = x
        for layer in self.layers:
            feat = F.dropout(self.relu(layer(feat, edge_index, edge_weight=edge_weight)),p=self.drop_p)
        return feat

class GCNPooling(nn.Module):
    def __init__(self, param, drop_p, device, sparse):
        super(GCNPooling, self).__init__()
        self.gcn = GCNEncoder(param, drop_p, device)
        self.softmax = nn.Softmax(dim=1)
        self.device = device
        self.sparse = sparse
    def forward(self,X_old, edge_index, edge_weight, A_old, Y_old, Z, use_sparse=False):
        """
        :param X_old:
        :param edge_index: 要求输入的邻接矩阵edge_index是对称的
        :param edge_weight:
        :param A_old: 等价于（edge_index, edge_weight）
        :param Y_old: 标签概率矩阵，防止因为argmax而导致类别丢失逐层传递
        :param Z: 该pooling层的经过gnn的embedding
        :param use_sparse: 第一层pooling要处理稀疏的原图
        :return:
        """
        S = self.softmax(self.gcn(X_old, edge_index, edge_weight=edge_weight))
        # # note 通过稀疏化指派矩阵达到稀疏图层的目的
        if self.sparse:
            # note top 1 sparse
            S = to_onehot(torch.argmax(S, dim=1),num_classes=S.shape[1],device=self.device)
            print('top1 sparse S')
            # # note top K sparse
            # k = 5
            # val, idx = torch.topk(S,k,1)
            # S = torch.zeros_like(S)
            # S.scatter(1,idx,val)
            # S = torch.softmax(S,dim=1)
            # print(f'top{k} sparse S')
        X_new = torch.matmul(S.T, Z)
        n_class = Y_old.shape[1]
        Y_new_prob = torch.softmax(torch.matmul(S.T, Y_old), dim=1)
        Y_new = to_onehot(torch.argmax(Y_new_prob, dim=1),n_class, device=self.device)
        # 第一层pooling，用的是原始图，是稀疏的
        if use_sparse:
            num_nodes_old = X_old.shape[0]
            row, col = edge_index[0,:], edge_index[1,:]
            tmp = spmm(index=torch.vstack([row, col]), value=edge_weight, m=num_nodes_old, n=num_nodes_old, matrix=S)
            A_new = torch.matmul(tmp.t(),S)
        # 第n层pooling, n>1, 用的是超图，是稠密带边权的
        else:
            A_new = torch.matmul(torch.matmul(S.T, A_old), S)
        return S, X_new, A_new, Y_new, Y_new_prob


class CondDiffPool(nn.Module):
    def __init__(self, conv_params, pool_params_s, pool_params_t,n_class,device, drop_p=0, classwise=False, share=0, sparse = 0):
        """

        :param conv_params: [[1st conv layer 参数],[],...]
        :param pool_params: [[1st pool layer 参数],[],...]
        :param n_class: 类别数，最终输出的cluster数
        """
        super(CondDiffPool, self).__init__()
        self.num_conv_blocks = len(conv_params)
        self.num_pool_blocks = len(pool_params_s)
        assert self.num_conv_blocks >= self.num_pool_blocks
        # 组装网络
        self.gnn_emb = []  # embedding gnn is shared between domains
        for i in range(self.num_conv_blocks):
            self.gnn_emb.append(GCNEncoder(conv_params[i], drop_p,device=device))
        self.gnn_emb = nn.ModuleList(self.gnn_emb)
        if not share:
            self.gnn_pool_src = []  # pooling gnn is seperated between domains
            self.gnn_pool_tgt = []
            for i in range(self.num_pool_blocks):
                self.gnn_pool_src.append(GCNPooling(pool_params_s[i],drop_p, device=device, sparse=sparse))
                self.gnn_pool_tgt.append(GCNPooling(pool_params_t[i],drop_p, device=device, sparse=sparse))
            self.gnn_pool_src = nn.ModuleList(self.gnn_pool_src)
            self.gnn_pool_tgt = nn.ModuleList(self.gnn_pool_tgt)
        else:
            self.gnn_pool = []
            for i in range(self.num_pool_blocks):
                self.gnn_pool.append(GCNPooling(pool_params_s[i], drop_p, device=device, sparse=sparse))
            self.gnn_pool = nn.ModuleList(self.gnn_pool)

        # 分类器。conv_params[0][-1]是第一个卷积模块的隐藏参数列表，取最后一个隐层的维度为分类器的输入维度。
        self.lin = nn.Linear(conv_params[0][-1][-1], n_class).to(device)
        # self.lin = nn.Sequential(nn.Linear(conv_params[0][-1][-1], 16),nn.ReLU(), nn.Linear(16, n_class)).to(device)
        self.classwise = classwise
        self.share = share
    def adj2coo(self, A):
        edge_weight = torch.squeeze(A.reshape(1,-1))
        row_elements = []
        col_elements = []
        n_node = A.shape[0]
        for i in range(n_node):
            row_elements.append(torch.Tensor([i]*n_node))
            col_elements.append(torch.arange(0,n_node))
        row = torch.hstack(row_elements)
        col = torch.hstack(col_elements)
        edge_index = torch.vstack([row,col]).long()
        return edge_index.to(self.device), edge_weight.to(self.device)

    def forward(self, x_s, edge_index_s, y_s, x_t, edge_index_t, y_t):
        self.device = x_s.device
        edge_weight_s = torch.ones(edge_index_s.shape[1]).to(self.device)
        edge_weight_t = torch.ones(edge_index_t.shape[1]).to(self.device)
        A_s, A_t = edge_index_s, edge_index_t  # 第一层pooling不需要原始图的邻接矩阵表示
        y_prob_s = y_s
        y_prob_t = y_t
        self.embedddings = []  # 存放每层输出的embedding zs zt。 xs xt相当于每层图的原始输入
        self.pooling_loss = []  # 存放pooling的loss
        self.y = []
        for i in range(self.num_conv_blocks):
            print(f"@conv block {i+1}")
            # message passing
            z_s = self.gnn_emb[i](x_s, edge_index_s, edge_weight_s)
            z_t = self.gnn_emb[i](x_t, edge_index_t, edge_weight_t)
            self.embedddings.append([z_s, z_t])
            # pseudo labeling if self.classwise and only @ first graph layer
            if self.classwise and i==0:
                y_t_pseudo = self.pseudo_label(z_s, y_s, z_t, y_t, edge_index_t, edge_weight_t)
                y_prob_t = y_t_pseudo
                self.y.append([y_s, y_t_pseudo])  # 存放每个图层的one-hot label matrix
                print(y_s.shape, y_t_pseudo.shape)
            # pooling
            if i<self.num_pool_blocks:
                if len(self.pooling_loss)<i+1: self.pooling_loss.append({})
                print(f"@pool block {i + 1}")
                use_sparse = True if i==0 else False
                if not self.share:
                    # NOTE 传入池化的是One-hot标签矩阵好，还是概率标签矩阵好？目前遵循公式传参是概率矩阵
                    S_s, x_s, A_s_new, y_s, y_prob_s_new = self.gnn_pool_src[i](X_old=x_s, edge_index=edge_index_s, edge_weight=edge_weight_s, A_old=A_s, Y_old=y_prob_s, Z=z_s, use_sparse=use_sparse)
                    S_t, x_t, A_t_new, y_t, y_prob_t_new = self.gnn_pool_tgt[i](X_old=x_t, edge_index=edge_index_t, edge_weight=edge_weight_t, A_old=A_t, Y_old=y_prob_t, Z=z_t, use_sparse=use_sparse)
                else:
                    S_s, x_s, A_s_new, y_s, y_prob_s_new = self.gnn_pool[i](X_old=x_s, edge_index=edge_index_s, edge_weight=edge_weight_s, A_old=A_s, Y_old=y_prob_s, Z=z_s, use_sparse=use_sparse)
                    S_t, x_t, A_t_new, y_t, y_prob_t_new = self.gnn_pool[i](X_old=x_t, edge_index=edge_index_t, edge_weight=edge_weight_t, A_old=A_t, Y_old=y_prob_t, Z=z_t, use_sparse=use_sparse)
                self.y.append([y_s, y_t])
                # pooling layer side loss
                # note 是否要/2
                self.pooling_loss[i]['ce']=(self.cluster_entropy(S_s)+self.cluster_entropy(S_t))/2
                self.pooling_loss[i]['prox'] = (self.proximity_loss(A_s, S_s)+self.proximity_loss(A_t, S_t))/2
                self.pooling_loss[i]['cce'] = (self.conditional_cluster_entropy(y_prob_s_new)+self.conditional_cluster_entropy(y_prob_t_new))/2
                self.pooling_loss[i]['lm'] = (self.label_matching(S_s, y_prob_s, y_prob_s_new)+self.label_matching(S_t, y_prob_t, y_prob_t_new))/2
                self.pooling_loss[i]['ls'] = (self.label_stable(S_s, y_prob_s, y_prob_s_new)+self.label_stable(S_t, y_prob_t, y_prob_t_new))/2
                # Update
                A_s, A_t = A_s_new, A_t_new
                y_prob_s, y_prob_t = y_prob_s_new, y_prob_t_new
                edge_index_s, edge_weight_s = self.adj2coo(A_s)
                edge_index_t, edge_weight_t = self.adj2coo(A_t)
                print(f"source {torch.sum(A_s>0).item()} edges, target {torch.sum(A_t>0).item()} edges")
        # 分类 使用第一个图层（原始图）的embedding，因为节点数随pooling变化
        pred_s = self.lin(self.embedddings[0][0])
        pred_t = self.lin(self.embedddings[0][1])
        pred = [pred_s, pred_t]
        return self.embedddings, pred, self.pooling_loss, self.y,self.lin(self.embedddings[1][0])

    def pseudo_label(self, z_s, y_s, z_t, y_t, edge_index_t, edge_weight_t):
        """return one-hot pseudo label matrix"""
        n_class = y_s.shape[1]
        entropy_lower_bound = 0.04
        gmmcluster = GMMClustering(num_class=n_class, device=self.device)
        # note 伪标签
        with torch.no_grad():
            pred_t = self.lin(z_t)
            _, tgt_indices = torch.max(torch.log_softmax(pred_t, dim=-1), dim=1)
            # try:
            #     src_center = get_emb_centers(z_s, y_s, n_class)
            #     tgt_samples = gmmcluster.forward(src_center, z_t, y_t, edge_index_t , edge_weight_t,
            #                                      smooth=False, smooth_r=0.5)
            #     # 通过熵选择用哪组伪标签
            #     if pred_entropy(tgt_samples['label']) > pred_entropy(tgt_indices) and pred_entropy(
            #             tgt_samples['label']) > entropy_lower_bound:
            #         tgt_y_pseudo = tgt_samples['label']
            #         print('use ours')
            #     else:
            #         tgt_y_pseudo = tgt_indices
            #     print(pred_entropy(tgt_samples['label']), pred_entropy(tgt_indices))
            #     # note pseudo label acc
            #     correct_pseudo = torch.sum(tgt_y_pseudo==torch.argmax(y_t, dim=1))
            #     correct_naive = torch.sum(tgt_indices==torch.argmax(y_t, dim=1))
            #     print(correct_pseudo/y_t.shape[0], correct_naive/y_t.shape[0])
            # except ValueError:
            #     tgt_y_pseudo = tgt_indices
            #     print('use naive')
        tgt_y_pseudo = tgt_indices
        tgt_y_pseudo = to_onehot(tgt_y_pseudo, n_class, self.device)
        return tgt_y_pseudo.to(self.device)

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)  # /len(kernel_val)

    def mmd(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target,
                                  kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss

    def simple_mmd(self, source, target):
        """
        不使用核函数，假设embedding直接在RKHS。 empirical MMD。
        :param source:
        :param target:
        :return:
        """
        source = torch.mean(source, dim=0)
        target = torch.mean(target, dim=0)
        return torch.norm(source - target)

    def simple_mmd_kernel(self, source, target):
        source = torch.mean(source, dim=0)
        target = torch.mean(target, dim=0)
        return torch.exp(-0.1*torch.norm(source - target))

    def classwise_simple_mmd(self, source, target, src_y, tgt_y):
        """
        如果某个类别目标域没有，就忽略该类
        :param source:
        :param target:
        :param src_y:  one-hot label matrix
        :param tgt_y:  one-hot label matrix
        :return:
        """
        mmd = 0.
        for c in range(src_y.shape[1]):
            src_idx = src_y[:, c].to(torch.bool)
            src = source[src_idx]
            tgt_idx = tgt_y[:, c].to(torch.bool)
            tgt = target[tgt_idx]
            # 对于目标域缺失的类别，直接忽略
            if not torch.isnan(self.simple_mmd(src, tgt)):
                mmd += self.simple_mmd(src, tgt)
            # print(c,simple_mmd(src,tgt))
        return mmd

    def classwise_simple_mmd_kernel(self, source, target, src_y, tgt_y):
        """
        如果某个类别目标域没有，就忽略该类
        :param source:
        :param target:
        :param src_y:  one-hot label matrix
        :param tgt_y:  one-hot label matrix
        :return:
        """
        mmd = 0.
        for c in range(src_y.shape[1]):
            src_idx = src_y[:, c].to(torch.bool)
            src = source[src_idx]
            tgt_idx = tgt_y[:, c].to(torch.bool)
            tgt = target[tgt_idx]
            # 对于目标域缺失的类别，直接忽略
            if not torch.isnan(self.simple_mmd(src, tgt)):
                mmd += self.simple_mmd_kernel(src, tgt)
            # print(c,simple_mmd(src,tgt))
        return mmd

    def proximity_loss(self, A, S, adj_hop=1):
        """pool neighboring nodes to the same cluster"""
        eps = 1e-7
        num_nodes = S.size()[0]
        pred_adj0 = torch.matmul(S, S.T)
        tmp = pred_adj0
        pred_adj = pred_adj0
        for adj_pow in range(adj_hop - 1):
            tmp = tmp @ pred_adj0
            pred_adj = pred_adj + tmp
        pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).to(self.device))
        pos_adj = torch.log(pred_adj + eps)
        neg_adj = torch.log(1 - pred_adj + eps)
        num_entries = num_nodes * num_nodes
        if A.shape[0]<A.shape[1]:  # 第一图层是COO格式的adj矩阵
            pos = torch.sum(pos_adj[A[0],A[1]])
            neg = torch.sum(neg_adj) - torch.sum(neg_adj[A[0],A[1]])
            link_loss = (-pos-neg)/float(num_entries)
        else:  # 第n层（n>1）是稠密的邻接矩阵
            link_loss = -A * torch.log(pred_adj + eps) - (1 - A) * torch.log(1 - pred_adj + eps)
            link_loss = torch.sum(link_loss) / float(num_entries)
        return link_loss

    def cluster_entropy(self,S):
        return entropy(S,reduction='mean')

    def conditional_cluster_entropy(self, Y):
        return entropy(Y, reduction='mean')

    def label_matching(self, S, Y_old, Y_new):
        """maximize the accordance between assignment and label-matching
            equal to minimize 1-accordance
        """
        n_node_old = S.shape[0]
        n_node_new = S.shape[1]
        label_matching_mat = torch.matmul(Y_old, Y_new.T)  # n_l * n_l+1
        label_matching_mat = to_onehot(torch.argmax(label_matching_mat, dim=1),num_classes=n_node_new, device=self.device)
        S = to_onehot(torch.argmax(S, dim=1),num_classes=S.shape[1], device=self.device)
        c = torch.sum(label_matching_mat*S)
        return 1-c/n_node_old

    def label_stable(self,S, Y_old, Y_new):
        """label stability before and after pooling"""
        n_class = Y_old.shape[1]
        label_stable_mat = torch.softmax(torch.matmul(torch.matmul(Y_old.T, S), Y_new), dim=1)
        pos = torch.mean(torch.diag(label_stable_mat))
        return 1-pos

