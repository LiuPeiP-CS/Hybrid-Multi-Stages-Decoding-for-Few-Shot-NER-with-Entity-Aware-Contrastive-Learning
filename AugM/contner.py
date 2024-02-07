#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/12/21 下午11:00
# @Author  : PeiP Liu
# @FileName: contner.py
# @Software: PyCharm

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

def nt_xent(loss, num, denom, temperature = 1):
    # loss: # 每个token与所有token(包含自己)计算的KL散度
    # num (loss_weights): # 同类别但不是自身的token标签处为1
    # denom (loss_mask): # 非自身位置为1，即为了区分自身位置
    #

    loss = torch.exp(loss/temperature) # 分子部分叠加的元素(token与每个正样本的相似度)
    cnts = torch.sum(num, dim = 1) # 每个token正样本的数量
    loss_num = torch.sum(loss * num, dim = 1) # 每个token与所有正样本的相似度之和，即对比学习分子部分
    loss_denom = torch.sum(loss * denom, dim = 1) # 每个token与所有样本(包括正负样本)的相似度之和，即对比学习分母部分
    # sanity check
    nonzero_indexes = torch.where(cnts > 0) # 对于所有token，只计算存在正样本的token相关的loss
    loss_num, loss_denom, cnts = loss_num[nonzero_indexes], loss_denom[nonzero_indexes], cnts[nonzero_indexes]

    loss_final = -torch.log2(loss_num) + torch.log2(loss_denom) + torch.log2(cnts) # 似乎不完全符合常规的对比学习公式？
    # loss_final = -torch.log2(loss_num) + torch.log2(loss_denom)
    # loss_final = loss_final*1.0/cnts.float()
    return loss_final # shape = [token_size, ]

def loss_kl(mu_i, sigma_i, mu_j, sigma_j, embed_dimension): # 以往的对比学习使用的相似度计算，该文章使用的散度计算
    # mu_i: filtered_embedding_mu　# 每个元素循环增强len次，即[X1, X1, X1, ..., X1(连续len个，同理于后续), X1, X2, X2, ...]，其中每个Xi都是一个向量
    # sigma_i: filtered_embedding_sigma　# 每个元素循环增强len次，即[X1, X1, X1, ..., X1(连续len个，同理于后续), X1, X2, X2, ...]，其中每个Xi都是一个向量
    # mu_j: repeated_output_embeddings_mu　# [X1, X2, X3, ... Xn, X1, X2, X3, ... Xn, ...]重复len次
    # sigma_j: , repeated_output_embeddings_sigma　# [X1, X2, X3, ... Xn, X1, X2, X3, ... Xn, ...]重复len次
    '''

    Calculates KL-divergence between two DIAGONAL Gaussians.
    Reference: https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians.
    Note: We calculated both directions of KL-divergence.
    '''
    sigma_ratio = sigma_j / sigma_i
    trace_fac = torch.sum(sigma_ratio, 1) # j元素对i元素的比率
    log_det = torch.sum(torch.log(sigma_ratio + 1e-14), dim=1)
    mu_diff_sq = torch.sum((mu_i - mu_j) ** 2 / sigma_i, dim=1)
    ij_kl = 0.5 * (trace_fac + mu_diff_sq - embed_dimension - log_det)
    sigma_ratio = sigma_i / sigma_j
    trace_fac = torch.sum(sigma_ratio, 1)
    log_det = torch.sum(torch.log(sigma_ratio + 1e-14), dim=1)
    mu_diff_sq = torch.sum((mu_j - mu_i) ** 2 / sigma_j, dim=1)
    ji_kl = 0.5 * (trace_fac + mu_diff_sq - embed_dimension - log_det)
    kl_d = 0.5 * (ij_kl + ji_kl)
    return kl_d # shape = [len*len, ]

def euclidean_distance(a, b, normalize=False):
    if normalize:
        a = F.normalize(a)
        b = F.normalize(b)
    logits = ((a - b) ** 2).sum(dim=1) # 逐元素相减后平方，然后在特征维度相加
    return logits # shape = [len*len, ]


def remove_irrelevant_tokens_for_loss(embedding_dimension, attention_mask, original_embedding_mu, original_embedding_sigma, labels):
    active_indices = attention_mask.view(-1) == 1 # 包含了[CLS],[SEP]
    active_indices = torch.where(active_indices == True)[0] # 真实token的位置[0,1,2,3,4,5,6...]

    output_embedding_mu = original_embedding_mu.view(-1, embedding_dimension)[active_indices] # (batch中所有有效字符的数量, out_dim)
    output_embedding_sigma = original_embedding_sigma.view(-1, embedding_dimension)[active_indices]
    labels_straightened = labels.view(-1)[active_indices] # 和output_embedding_mu的0维一样大

    # remove indices with negative labels only

    nonneg_indices = torch.where(labels_straightened >= 0)[0] # 正式标签包括0，意在消除[CLS],[SEP]的影响
    output_embedding_mu = output_embedding_mu[nonneg_indices]
    output_embedding_sigma = output_embedding_sigma[nonneg_indices]
    labels_straightened = labels_straightened[nonneg_indices]

    return output_embedding_mu, output_embedding_sigma, labels_straightened # 纯粹的字符向量以及对应的标签，成为(X1+X2+X3+...Xn, emb_dim)形式


def calculate_KL_or_euclidean(embedding_dimension, attention_mask, original_embedding_mu, original_embedding_sigma, labels,
                              consider_mutual_O=False, loss_type=None):

    # we will create embedding pairs in following manner
    # filtered_embedding | embedding ||| filtered_labels | labels
    # repeat_interleave |            ||| repeat_interleave |
    #                   | repeat     |||                   | repeat
    # extract only active parts that does not contain any paddings

    output_embedding_mu, output_embedding_sigma, labels_straightened = remove_irrelevant_tokens_for_loss(embedding_dimension, attention_mask, original_embedding_mu, original_embedding_sigma, labels)

    # remove indices with zero labels, that is "O" classes, 即0是非实体的标签
    if not consider_mutual_O:
        filter_indices = torch.where(labels_straightened > 0)[0] # 存在实际实体的标签，标签0表示无实体标签
        filtered_embedding_mu = output_embedding_mu[filter_indices]
        filtered_embedding_sigma = output_embedding_sigma[filter_indices]
        filtered_labels = labels_straightened[filter_indices]
    else: # 本程序实际执行此处
        filtered_embedding_mu = output_embedding_mu
        filtered_embedding_sigma = output_embedding_sigma
        filtered_labels = labels_straightened

    filtered_instances_nos = len(filtered_labels)

    # repeat interleave
    filtered_embedding_mu = torch.repeat_interleave(filtered_embedding_mu, len(output_embedding_mu), dim=0) # 每个元素循环增强len次，即[X1, X1, X1, ..., X1(连续len个，同理于后续), X1, X2, X2, ...]，其中每个Xi都是一个向量
    filtered_embedding_sigma = torch.repeat_interleave(filtered_embedding_sigma, len(output_embedding_sigma),dim=0)
    filtered_labels = torch.repeat_interleave(filtered_labels, len(output_embedding_mu), dim=0) # 单token标签的连续扩充

    # only repeat
    repeated_output_embeddings_mu = output_embedding_mu.repeat(filtered_instances_nos, 1)
    repeated_output_embeddings_sigma = output_embedding_sigma.repeat(filtered_instances_nos, 1)
    repeated_labels = labels_straightened.repeat(filtered_instances_nos) # [X1, X2, X3, ... Xn, X1, X2, X3, ... Xn, ...]重复len次

    # avoid losses with own self
    loss_mask = torch.all(filtered_embedding_mu != repeated_output_embeddings_mu, dim=-1).int() # 非自身位置为1，即为了区分自身位置
    loss_weights = (filtered_labels == repeated_labels).int() # 同类型token的标签比较。每个token的标签与所有token的标签比较(包括自身)
    loss_weights = loss_weights * loss_mask # 同类别但不是自身的token标签处为1

    #ensure that the vector sizes are of filtered_instances_nos * filtered_instances_nos
    assert len(repeated_labels) == (filtered_instances_nos * filtered_instances_nos), "dimension is not of square shape."

    if loss_type == "euclidean":
        loss = -euclidean_distance(filtered_embedding_mu, repeated_output_embeddings_mu, normalize=True)

    elif loss_type == "KL":  # KL_divergence
        loss = -loss_kl(filtered_embedding_mu, filtered_embedding_sigma,
                            repeated_output_embeddings_mu, repeated_output_embeddings_sigma,
                            embed_dimension=embedding_dimension)

    else:
        raise Exception("unknown loss")

    # reshape the loss, loss_weight, and loss_mask
    loss = loss.view(filtered_instances_nos, filtered_instances_nos) # 每个token与所有token(包含自己)计算的KL散度
    loss_mask = loss_mask.view(filtered_instances_nos, filtered_instances_nos) # 非自身位置为1，即为了区分自身位置
    loss_weights = loss_weights.view(filtered_instances_nos, filtered_instances_nos) # 同类别但不是自身的token标签处为1

    loss_final = nt_xent(loss, loss_weights, loss_mask, temperature = 1)
    return torch.mean(loss_final) # 所有token的对比学习loss求平均


class ContrastiveLearning(nn.Module): # modified the original huggingface BertForTokenClassification to incorporate gaussian
    def __init__(self, hidden_size=768, embedding_dimension=128, hidden_dropout_prob=0.3):
        super(ContrastiveLearning, self).__init__()
        # 初始化内容应该是hidden_size和embedding_dims
        # super().__init__(config)
        # self.num_labels = config.num_labels
        # self.embedding_dimension = config.task_specific_params['embedding_dimension']

        self.embedding_dimension = embedding_dimension

        # self.bert = BertModel(config)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        # self.projection = nn.Sequential(
        #     nn.Linear(config.hidden_size, self.embedding_dimension + (config.hidden_size - self.embedding_dimension) // 2)
        # )

        self.output_embedder_mu = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_dimension))

        self.output_embedder_sigma = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_dimension))

    def forward(
            self,
            ent_embeddings, # 实体的向量表示，(batch_size, max_ent_num, hidden_dim)
            ent_type_ids, # 实体的类型，(batch_size, max_ent_num)
            ent_mask, # 实体在序列中的mask表示，(batch_size, max_ent_num)
            loss_type: str, # 在进行N-way 1-shot时，采用loss_type == "euclidean"；其他时候，采用loss_type == "KL"
            consider_mutual_O=False # 默认值就是该值
    ): # 输入应该是BERT输出后的结果，以entity-span的形式得到的embeddings

        sequence_output = self.dropout(ent_embeddings) # shape=[batch_size, seq_len, out_dim]
        original_embedding_mu = ((self.output_embedder_mu((sequence_output)))) # shape=[batch_size, seq_len, 32]
        original_embedding_sigma = (F.elu(self.output_embedder_sigma((sequence_output)))) + 1 + 1e-14 # # shape=[batch_size, seq_len, 32]，另一种线性变换


        loss = calculate_KL_or_euclidean(self.embedding_dimension, ent_mask, original_embedding_mu,
                                                 original_embedding_sigma, ent_type_ids, consider_mutual_O,
                                                 loss_type=loss_type)
        return loss
