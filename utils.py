import datetime
import dgl
import numpy as np
import torch
from dgl.data.utils import _get_dgl_url, download, get_download_dir, load_graphs
from scipy import io as sio, sparse
from sklearn.metrics import f1_score
import torch.nn.functional as F
import math
import torch.utils.data as data
import scipy.sparse as sp
import os
import pandas as pd



"""
LLM related
"""

def get_retrieve_score(his_jobs_pos, his_comps_pos, all_jobs_emb, all_comps_emb, all_jobs_w, all_comps_w):
    # concate all_jobs_emb & all_comp_emb and update the his_jobs/comps_pos in the concatenated matrix
    all_nodes_emb = torch.cat([all_jobs_emb, all_comps_emb], dim=0)
    num_all_nodes_candi_g = all_nodes_emb.shape[0]
    new_jobs_pos = his_jobs_pos
    new_comp_pos = [pos+all_jobs_emb.shape[0] for pos in his_comps_pos]
    query_pos = new_jobs_pos + new_comp_pos
    num_all_queries = len(query_pos)
    cos_tensor = torch.zeros(num_all_queries, num_all_nodes_candi_g) # todo: check: here we consider each node can be both query and candidate doc
    for r_id, q_id in enumerate(query_pos):
        for d_id in range(num_all_nodes_candi_g): 
            query_emb = all_nodes_emb[q_id]
            doc_emb = all_nodes_emb[d_id]
            cos_tensor[r_id, d_id] = F.cosine_similarity(query_emb, doc_emb, dim=0) # if q_id == d_id, then cos_sim == 1

    # get all query_emb's entropy
    entropy_tensor = torch.zeros(num_all_queries) 
    for q_id in range(num_all_queries):
        prob_q = ((1 + cos_tensor[q_id]) / 2) / torch.sum((1 + cos_tensor[q_id]) / 2) # 1-dim tensor with length num_all_nodes_candi_g
        entropy_tensor[q_id] = -1 * torch.sum(torch.tensor([p*math.log(p) for p in prob_q]))

    avg_entropy = torch.mean(entropy_tensor)
    scores_on_q = torch.zeros(num_all_queries, num_all_nodes_candi_g)
    for q_id in range(num_all_queries):
        scores_on_q[q_id] = torch.exp(((1 + cos_tensor[q_id]) / 2) * torch.sigmoid(avg_entropy - entropy_tensor[q_id]))
    
    # multiply candidate nodes weights
    all_nodes_w = torch.cat([all_jobs_w, all_comps_w], dim=0) # job + comp [num_all_nodes, 1]
    final_scores = torch.cat([torch.matmul(scores_on_q[q_id].to(all_nodes_w.device), all_nodes_w) for q_id in range(num_all_queries)], dim=0).reshape(-1) # length: num_all_queries

    
    final_scores = torch.log(final_scores)
    return torch.sum(final_scores)

"""
GNN related
"""


def split_edges(graph, etype, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    # get positive edges
    u, v = graph.edges(etype=etype)
    eids = np.arange(graph.num_edges(etype=etype))
    eids = np.random.permutation(eids)

    num_edges = len(eids)
    train_size = int(num_edges * train_ratio)
    val_size = int(num_edges * val_ratio)
    test_size = num_edges - train_size - val_size

    train_pos_eids = eids[:train_size]
    val_pos_eids = eids[train_size:train_size + val_size]
    test_pos_eids = eids[train_size + val_size:]

    train_pos_u, train_pos_v = u[train_pos_eids], v[train_pos_eids]
    val_pos_u, val_pos_v = u[val_pos_eids], v[val_pos_eids]
    test_pos_u, test_pos_v = u[test_pos_eids], v[test_pos_eids]

    # build negative edges
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))

  
    adj_neg = 1 - adj.todense()
    neg_u, neg_v = np.where(adj_neg != 0) 

    # Randomly sample an equal number of edges from negative samples
    neg_eids = np.random.choice(len(neg_u), num_edges, replace=False)
    train_neg_u, train_neg_v = neg_u[neg_eids[:train_size]], neg_v[neg_eids[:train_size]]
    val_neg_u, val_neg_v = neg_u[neg_eids[train_size:train_size + val_size]], neg_v[neg_eids[train_size:train_size + val_size]]
    test_neg_u, test_neg_v = neg_u[neg_eids[train_size + val_size:]], neg_v[neg_eids[train_size + val_size:]]

    
    splits = {
        'train_pos': (train_pos_u, train_pos_v),
        'val_pos': (val_pos_u, val_pos_v),
        'test_pos': (test_pos_u, test_pos_v),
        'train_neg': (train_neg_u, train_neg_v),
        'val_neg': (val_neg_u, val_neg_v),
        'test_neg': (test_neg_u, test_neg_v),
    }
    eid_dict_split = {
        'train': train_pos_eids,
        'val': val_pos_eids,
        'test': test_pos_eids
    }
    # Return the split edge IDs for later use in dgl.remove_edges() to construct the training KG
    return splits, eid_dict_split



def get_edge_dict(splits_b, splits_h, splits_t2j, splits_t2c, graph_type):
    # param graph_type should be: "train/test/val_pos/neg"
    graph_data = {
        ('job', 'belongsto', 'company'): splits_b[graph_type],
        ('company', 'has', 'job'): splits_h[graph_type],
        ('job', 'transtojob', 'job'): splits_t2j[graph_type],
        ('company', 'transtocomp', 'company'): splits_t2c[graph_type]
    }
    return graph_data

def build_hetero_sg(career_kg, splits_b, splits_h, splits_t2j, splits_t2c, graph_type):
    edge_dict = {
        ('job', 'belongsto', 'company'): splits_b[graph_type],
        ('company', 'has', 'job'): splits_h[graph_type],
        ('job', 'transtojob', 'job'): splits_t2j[graph_type],
        ('company', 'transtocomp', 'company'): splits_t2c[graph_type]
    }
    subgraph_data = {}
    for etype, (src, dst) in edge_dict.items():
        subgraph_data[etype] = (src, dst)
    
    # construct subgraph
    subgraph = dgl.heterograph(
        subgraph_data,
        num_nodes_dict={ntype: career_kg.num_nodes(ntype) for ntype in career_kg.ntypes}
    )

    # 复制节点特征
    for ntype in career_kg.ntypes:
        for feat_name in career_kg.nodes[ntype].data.keys():
            subgraph.nodes[ntype].data[feat_name] = career_kg.nodes[ntype].data[feat_name]

    
    return subgraph

def load_all_graphs(org_kg_dir):
    career_kg, _ = load_graphs(org_kg_dir)
    career_kg = career_kg[0]
    splits_b, eid_dict_split_b = split_edges(career_kg, ('job', 'belongsto', 'company'))
    splits_h, eid_dict_split_h = split_edges(career_kg, ('company', 'has', 'job'))
    splits_t2j, eid_dict_split_t2j = split_edges(career_kg, ('job', 'transtojob', 'job'))
    splits_t2c, eid_dict_split_t2c = split_edges(career_kg, ('company', 'transtocomp', 'company'))

    split_eids_dict = {
        ('job', 'belongsto', 'company'): eid_dict_split_b,
        ('company', 'has', 'job'): eid_dict_split_h,
        ('job', 'transtojob', 'job'): eid_dict_split_t2j,
        ('company', 'transtocomp', 'company'): eid_dict_split_t2c
    }
    graph_types_list = ["train_pos", "train_neg", "test_pos", "test_neg", "val_pos", "val_neg"]
    all_pos_neg_graphs_list = {}
    for g_type in graph_types_list:
        all_pos_neg_graphs_list[g_type] = build_hetero_sg(career_kg, splits_b, splits_h, splits_t2j, splits_t2c, g_type)
    train_career_kg = career_kg
    for etype in career_kg.canonical_etypes:
        rmv_eids = np.concatenate((split_eids_dict[etype]['val'], split_eids_dict[etype]['test']))
        train_career_kg = dgl.remove_edges(train_career_kg, rmv_eids, etype=etype)
    return train_career_kg, all_pos_neg_graphs_list


class EarlyStopping_GNN_train(object):
    def __init__(self, patience=10, data_set_type='fin', cpt_dir="Multi_HAN_results", seed=0):
        dt = datetime.datetime.now()
        
        self.filename = os.path.join(cpt_dir, "early_stop_cp_{}_{}_{:02d}-{:02d}-{:02d}_{}.pth".format(
            str(seed), dt.date(), dt.hour, dt.minute, dt.second, data_set_type
        ))
        self.patience = patience
        self.counter = 0
        self.best_auc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, auc, model):
        if self.best_loss is None:
            self.best_auc = auc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (auc < self.best_auc):
            self.counter += 1
            print(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (auc >= self.best_auc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_auc = np.max((auc, self.best_auc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))