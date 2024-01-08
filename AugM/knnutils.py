#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/12/21 下午11:01
# @Author  : PeiP Liu
# @FileName: knnutils.py
# @Software: PyCharm

import numpy as np
import torch
from torch.nn import functional as F


def build_datastore(sup_ents_emb, sup_ents_labels, sup_ents_mask):
    # 面向test测试集的support支持集，构建{emb: ent_label}的数据库
    token_sum = sum(int(x.sum(dim=-1).cpu()) for x in sup_ents_mask)
    hidden_size = sup_ents_emb.shape[-1]

    data_store_key_in_memory = np.zeros((token_sum, hidden_size), dtype=np.float32)
    data_store_val_in_memory = np.zeros((token_sum,), dtype=np.int32)

    r_features = sup_ents_emb.reshape(-1, hidden_size) # (ent_num*sent*num, hid_dim)
    r_mask = sup_ents_mask.reshape(-1).bool()
    r_labels = sup_ents_labels.reshape(-1)

    labels = torch.masked_select(r_labels, r_mask).cpu().numpy() # 此处直接使用r_labels[r_mask].cpu().numpy()也可以实现
    mask = r_mask.unsqueeze(1).repeat(1, hidden_size) # 同理可以由r_features[r_mask].view(-1, hidden_size).cpu()实现
    features = torch.masked_select(r_features, mask).view(-1, hidden_size).cpu()
    np_features = features.numpy().astype(np.float32)

    data_store_key_in_memory[:] = np_features
    data_store_val_in_memory[:] = labels

    return {'keys': data_store_key_in_memory, 'vals': data_store_val_in_memory}

def ModelAKNN4prob(sup_Datastore, testEntEmb, ModelLogits, args):
    # 此处为KNNNER的核心
    keys = torch.from_numpy(sup_Datastore['keys']).to(args.device)
    vals = torch.from_numpy(sup_Datastore['vals']).to(args.device)
    link_temperature = torch.tensor(args.link_temperature).to(args.device)
    link_ratio = torch.tensor(args.link_ratio).to(args.device)


    """input logits should in the shape [seq_num, seq_len, num_labels]"""
    probabilities = F.softmax(ModelLogits, dim=2)  # shape of [seq_num, seq_len, num_labels]
    num_labels = probabilities.shape[-1]

    seq_num = testEntEmb.shape[0]
    max_ent_num = testEntEmb.shape[1]
    hidden_size = testEntEmb.shape[-1]
    token_num = keys.shape[0]

    # cosine similarity
    knn_feats = keys.transpose(0, 1)  # [hid_dim, token_num]
    testEntEmb = testEntEmb.view(-1, hidden_size)  # [seq_num*max_ent_num, hid_dim]
    sim = torch.mm(testEntEmb, knn_feats)  # [seq_num*max_ent_num, token_num]
    norm_1 = (knn_feats ** 2).sum(dim=0, keepdim=True).sqrt()  # [1, token_num]
    norm_2 = (testEntEmb ** 2).sum(dim=1, keepdim=True).sqrt()  # [seq_num*max_ent_num, 1]
    scores = (sim / (norm_1 + 1e-10) / (norm_2 + 1e-10)).view(seq_num, max_ent_num, -1)  # [seq_num, max_ent_num, token_num]
    knn_labels = vals.view(1, 1, token_num).expand(seq_num, max_ent_num, token_num)  # [seq_num, max_ent_num, token_num]

    if (args.topk != -1 and scores.shape[-1] > args.topk):
        topk_scores, topk_idxs = torch.topk(scores, dim=-1, k=args.topk)  # [seq_num, max_ent_num, topk]
        scores = topk_scores
        knn_labels = knn_labels.gather(dim=-1, index=topk_idxs)  # [seq_num, max_ent_num, topk]

    sim_probs = torch.softmax(scores / link_temperature, dim=-1)  # [seq_num, max_ent_num, token_num]

    knn_probabilities = torch.zeros_like(sim_probs[:, :, 0]).unsqueeze(-1).repeat([1, 1, num_labels])  # [seq_num, max_ent_num, num_labels]
    knn_probabilities = knn_probabilities.scatter_add(dim=2, index=knn_labels,
                                                      src=sim_probs)  # [seq_num, max_ent_num, num_labels]

    probabilities = link_ratio * knn_probabilities + (1 - link_ratio) * probabilities

    return probabilities




