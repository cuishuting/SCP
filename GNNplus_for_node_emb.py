import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GATConv
from dgl.data.utils import load_graphs
from sklearn.metrics import roc_auc_score
from utils import EarlyStopping_GNN_train, load_all_graphs
import random
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']
        
 
# define multi-meta-paths HAN model as GNN
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        # z: (N, M, D * K)
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        
        return (beta * z).sum(1)  # (N, D * K)
    
class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : DGLGraph
        The heterogeneous graph
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, comp_meta_paths, job_meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.comp_gat_layers = nn.ModuleList()
        self.job_gat_layers = nn.ModuleList()
        for c in range(len(comp_meta_paths)):
            self.comp_gat_layers.append(
                GATConv(
                    in_size,
                    out_size,
                    layer_num_heads, # 8
                    dropout,
                    dropout,
                    activation=F.elu,
                    allow_zero_in_degree=True,
                )
            )
        for j in range(len(job_meta_paths)):
            self.job_gat_layers.append(
                GATConv(
                    in_size,
                    out_size,
                    layer_num_heads, # 8
                    dropout,
                    dropout,
                    activation=F.elu,
                    allow_zero_in_degree=True,
                )
            )
        self.semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads
        )
        self.comp_meta_paths = list(tuple(meta_path) for meta_path in comp_meta_paths)
        self.job_meta_paths = list(tuple(meta_path) for meta_path in job_meta_paths)

        self._cached_graph_comp = None
        self._cached_coalesced_graph_comp = {}

        self._cached_graph_job = None
        self._cached_coalesced_graph_job = {}

    def forward(self, g, h):
        comp_semantic_embeddings = []
        job_semantic_embeddings = []
        comp_f = h['company'] # [N, 384]
        job_f = h['job'] # [N, 384]
        
        # `self._cached_graph is not g` is used to re-obtain `self._cached_coalesced_graph` when training on multiple subgraphs
        if self._cached_graph_comp is None or self._cached_graph_comp is not g:
            self._cached_graph_comp = g
            self._cached_coalesced_graph_comp.clear()
            for meta_path in self.comp_meta_paths:
                self._cached_coalesced_graph_comp[
                    meta_path
                ] = dgl.metapath_reachable_graph(g, meta_path)
        
        if self._cached_graph_job is None or self._cached_graph_job is not g:
            self._cached_graph_job = g
            self._cached_coalesced_graph_job.clear()
            for meta_path in self.job_meta_paths:
                self._cached_coalesced_graph_job[
                    meta_path
                ] = dgl.metapath_reachable_graph(g, meta_path)

        for i, comp_meta_path in enumerate(self.comp_meta_paths):
            new_g = self._cached_coalesced_graph_comp[comp_meta_path]
            comp_semantic_embeddings.append(self.comp_gat_layers[i](new_g, comp_f).flatten(1)) # [N, D * K]

        for i, job_meta_path in enumerate(self.job_meta_paths):
            new_g = self._cached_coalesced_graph_job[job_meta_path]
            job_semantic_embeddings.append(self.job_gat_layers[i](new_g, job_f).flatten(1))
        
        comp_semantic_embeddings = torch.stack(comp_semantic_embeddings, dim=1) # (N, c_M, D * K)
        job_semantic_embeddings = torch.stack(job_semantic_embeddings, dim=1) # (N, j_M, D * K)

        
        return {
            "company": self.semantic_attention(comp_semantic_embeddings), # (N, D * K)
            "job": self.semantic_attention(job_semantic_embeddings) # (N, D * K)
        }  

class HAN(nn.Module):
    def __init__(
        self, comp_meta_paths, job_meta_paths, in_size, hidden_size, num_heads, dropout
    ):
        super(HAN, self).__init__()
        # args['num_heads']:[8]
        self.layers = nn.ModuleList()
        self.layers.append(
            # comp_meta_paths, job_meta_paths, in_size, out_size, layer_num_heads, dropout
            HANLayer(comp_meta_paths, job_meta_paths, in_size, hidden_size, num_heads[0], dropout)
        )
        for l in range(1, len(num_heads)):
            self.layers.append(
                HANLayer(
                    comp_meta_paths,
                    job_meta_paths,
                    hidden_size * num_heads[l - 1],
                    hidden_size,
                    num_heads[l],
                    dropout,
                )
            )
        # self.predict = nn.Linear(hidden_size * num_heads[-1], out_size) # out_size： num_classes # now we focus on link prediction task not the node classification task for node representation learning

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)

        return h
    

class Model_multi_mpath(nn.Module):
    def __init__(self, comp_meta_paths, job_meta_paths, in_features, hidden_features, num_heads, dropout):
        super().__init__()
        # HAN params: comp_meta_paths, job_meta_paths, in_size, hidden_size, num_heads, dropout
        self.multi_metapath_HAN = HAN(comp_meta_paths, job_meta_paths, in_features, hidden_features, num_heads, dropout)
        self.pred = HeteroDotProductPredictor()
    def forward(self, g, neg_g, x, etype):
        h = self.multi_metapath_HAN(g, x)
        return self.pred(g, h, etype), self.pred(neg_g, h, etype), h
    
    
def compute_loss(pos_score, neg_score):
    #  Binary Cross Entropy
    # pos_score, neg_score: [N, 1]
    scores = torch.cat([pos_score, neg_score], dim=0)
    labels = torch.cat(
        [torch.ones(pos_score.shape[0], 1, device=pos_score.device), torch.zeros(neg_score.shape[0], 1, device=neg_score.device)], dim=0
    )
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score.cpu(), neg_score.cpu()]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).numpy()
    return roc_auc_score(labels, scores)
    

def evaluate(model, val_pos_g, val_neg_g, nodes_f):
    auc_avg = 0
    eval_loss_avg = 0
    model.eval()
    with torch.no_grad():
        for etype in val_pos_g.canonical_etypes:
            pos_score, neg_score, _ = model(val_pos_g, val_neg_g, nodes_f, etype)
            eval_loss_avg += compute_loss(pos_score, neg_score)
            auc_avg += compute_auc(pos_score, neg_score)
        eval_loss_avg = eval_loss_avg / len(val_pos_g.etypes)
        auc_avg = auc_avg / len(val_pos_g.etypes)
    return eval_loss_avg, auc_avg



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("GNNplus")
    parser.add_argument("--KG_dir", type=str)
    parser.add_argument("--data_type", type=str, default='fin')
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--num_heads", type=int)
    args = parser.parse_args()

    num_epochs = 500
    device = "cuda:0"
    model = Model_multi_mpath(
        comp_meta_paths=[['has', 'belongsto'], ["has", "transtojob", "belongsto"]], 
        job_meta_paths=[['belongsto', 'has'], ['belongsto', 'transtocomp', 'has']],
        in_features=384,
        hidden_features=args.hidden_dim,
        num_heads=[args.num_heads], # [8],
        dropout=args.dropout #0.2
    ).to(device) # forward() input should be: g, neg_g, x, etype

    train_graph, all_pos_neg_graphs_dict = load_all_graphs(args.KG_dir) # "/data/cst/talent/career_KG/career_KG.bin"
    train_graph = train_graph.to(device)
    
    input_nodes_f = {
        "company": train_graph.nodes['company'].data['name_emb'].to(device),
        "job": train_graph.nodes['job'].data['title_emb'].to(device)
    }


    train_neg_g = all_pos_neg_graphs_dict["train_neg"].to(device)
    val_pos_g = all_pos_neg_graphs_dict["val_pos"].to(device)
    val_neg_g = all_pos_neg_graphs_dict["val_neg"].to(device)
    test_pos_g = all_pos_neg_graphs_dict["test_pos"].to(device)
    test_neg_g = all_pos_neg_graphs_dict["test_neg"].to(device)


    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    stopper = EarlyStopping_GNN_train(data_set_type=args.data_type)
    for epoch in range(num_epochs):
        model.train()
        loss = 0
        for etype in train_graph.canonical_etypes:
            pos_score, neg_score, _ = model(train_graph, train_neg_g, input_nodes_f, etype)
            loss += compute_loss(pos_score, neg_score)
        loss = loss / len(train_graph.etypes)

        opt.zero_grad()
        loss.backward()
        opt.step()

        print(f"Epoch {epoch+1}: training loss: {loss}")

        if (epoch+1) % 10 == 0:
            eval_loss_avg, auc_avg = evaluate(model, val_pos_g, val_neg_g, input_nodes_f)
            print(f"check val auc:{auc_avg}, check val loss:{eval_loss_avg}")
            early_stop = stopper.step(eval_loss_avg.data.item(), auc_avg, model)
            if early_stop:
                break

    stopper.load_checkpoint(model)
    test_loss_avg, test_auc_avg = evaluate(model, test_pos_g, test_neg_g, input_nodes_f)
    print(f"Final Eval performance: Test auc_avg: {test_auc_avg}; Test loss_avg:{test_loss_avg}")
