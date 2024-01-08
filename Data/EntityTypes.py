#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/11/17 下午11:34
# @Author  : PeiP Liu
# @FileName: preprocessor.py
# @Software: PyCharm

import bisect
import json
import logging

import numpy
import torch
import torch.nn as nn
import numpy as np

from transformers import BertModel, BertTokenizer
from Data.utils import load_file

logger = logging.getLogger(__file__) # 构建日志对象


class EntityTypes(object):
    def __init__(self, types_path, negative_mode): # No modification required
        self.types = {}
        self.types_map = {}
        self.O_id = 0
        self.types_embedding = None
        self.negative_mode = negative_mode # 默认值为batch
        self.load_entity_types(types_path)

    def load_entity_types(self, types_path): # No modification required
        """
        :param types_path: 保存类型的文件地址
        :return: 构建type4id的类型字典
        """
        self.types = load_file(types_path, 'json') # 类型字典, {XX: [], XXX: [], ...}

        types_list = sorted([jj for ii in self.types.values() for jj in ii]) # 包含所有类型的列表，并经过了排序
        self.types_list = types_list

        self.types_map = {jj: ii for ii, jj in enumerate(types_list)} # type4id
        self.O_id = self.types_map['O'] # 标签O的编号
        logger.info('Load %d entity types from %s.', len(self.types_list), types_path)

    def building_types_embedding(self, bert_model: str, do_lower_case: bool, device, type_mode: str, init_type_embedding_from_bert: bool):
        """
        :param bert_model: 预训练好的bert模型
        :param do_lower_case: True, 是否需要字母小写
        :param device: 选择什么样的设备
        :param type_mode: 'cls'，表示获取类型(描述)向量的方式
        :param init_type_embedding_from_bert: True，从bert中进行向量的初始化获取
        :return: 类型的初始化向量表示
        """
        if init_type_embedding_from_bert: # 传入数据是True
            tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

            types_tokens_id_list = [
                [tokenizer.cls_token_id] + # 加cls的token id
                tokenizer.convert_tokens_to_ids(tokenizer.tokenize(type_tokens)) + # 类型描述的token id
                [tokenizer.sep_token_id]
                for type_tokens in self.types_list
            ] # [type1_tokens_id, type2_tokens_id, ...]

            types_tokens_max_len = max([len(tokens) for tokens in types_tokens_id_list]) # 获取类型描述的最大长度
            mask = [[1] * len(types_tokens_ids) + [0] * (types_tokens_max_len-len(types_tokens_ids)) for types_tokens_ids in types_tokens_id_list]
            ids = [
                types_tokens_ids + [tokenizer.pad_token_id] * (types_tokens_max_len-len(types_tokens_ids))
                for types_tokens_ids in types_tokens_id_list
            ] # 每个类型描述转换成为token_ids，所有的类型描述构造成为列表。[token_ids1, token_ids2, ...]
            mask = torch.tensor(mask, dtype=torch.long).to(device)
            ids = torch.tensor(ids, dtype=torch.long).to(device) # len(type_list) * types_tokens_max_len

            bert_model = BertModel.from_pretrained(bert_model).to(device) # bert模型的
            outputs = bert_model(ids, mask) # bert模型的输出，包括last_hid_state, pooler_output,hidden_states, attentions

        else:
            # 如果不需要bert模型进行初始化，那么就直接随机化
            outputs = [0, torch.rand((len(self.types_list), 768)).to(device)] # 随机初始化，第一个0是为了后面的操作方便

        if type_mode != 'cls':
            self.types_embedding = nn.Embedding(*outputs[1].shape).to(device) # *outputs[1].shape即(len(self.types_list), 768),构建了一个nn.Embedding对象
        else:
            self.types_embedding = nn.Parameter(outputs[1], requires_grad=False) # 载入bert得到的向量数据

        logger.info('The type embedding has been built !')

    def generate_negative_types(self, ent_type_ids, task_types_ids, negative_types_number):
        """
        :param ent_type_ids: 一个句子序列中的实际实体类型列表
        :param task_types_ids: 一个句子所在的任务中，所有的types，即task['types']
        :param negative_types_number:  样本的负类型数量, K-1
        :return: 对一个句子序列中的实际实体类型，补充负样本实体类型进行扰动
        """
        task_types_ids = set(task_types_ids) # 将原始的列表转换成为set，便于求差值
        task_types_num = len(task_types_ids) # 该任务中，所有的types的数量

        ent_num = len(ent_type_ids) # 实体的类型列表，实际上也是实体的数量
        data = np.zeros((ent_num, 1 + negative_types_number), np.int)

        if self.negative_mode == 'batch':
            if negative_types_number > task_types_num:
                other_types = list(set(range(len(self.types_map))) - task_types_ids - set(self.O_id)) # 不在该任务中的其他的所有实体类型
                other_num = negative_types_number - task_types_num
            else:
                other_types, other_num = [], 0

            if negative_types_number > task_types_num - 1:
                b_size = task_types_num-1 # b_size的含义在于优先在本task中选择负样本类型
                o_set = [self.O_id]
            else:
                b_size = negative_types_number
                o_set = []

            for idx, ent_type_id in enumerate(ent_type_ids):
                data[idx][0] = ent_type_id # 作为实际的实体类型标签
                data[idx][1:] = np.concatenate([
                    np.random.choice(list(task_types_ids - set([ent_type_id])), b_size, False), # 集合的remove()是在原始变量的基础上进行操作的，因此不会有返回值。
                    o_set,
                    np.random.choice(other_types, other_num, False),
                ])

        return data # shape = (ent_num, negative_types_number + 1)

    def update_type_embedding(self, ent_out, ent_type_ids, ent_type_mask):
        """
        以一个task为单位进行函数操作.
        该函数的作用是使用基于训练数据(包括support和query数据)对实体的类型进行原始网络向量的计算
        :param ent_out: entity-span的特征表示，shape=(seq_num, max_ent, hidden_dim)
        :param ent_type_ids: # max_ent_num * (1+negative_type_num), 每个元素为[ent_type_id, negative_type1, negative_type_id2, ... negative_num])
        :param ent_type_mask: # shape = max_ent_num * (1+negative_type_num)，其中存在实体的前ent_num * (1+negative_type_num)为1，其余部分为0
        :return:
        """
        ent_true_type_ids = ent_type_ids[:, :, 0][ent_type_mask[:, :, 0] == 1] # 真实类型的id，排除了padding
        all_hiddens = ent_out[ent_type_mask[:, :, 0] == 1] # (entity_num_in_all_seq,  hidden_size)，entity_num_in_all_seq表示所有seq中实体数量
        type_set = set(ent_true_type_ids.detach().cpu().numpy()) # support或者query中涉及到的task中的type信息
        for type in type_set:
            self.types_embedding.data[type] = all_hiddens[ent_true_type_ids == type].mean(0) # 使用同类型实体的特征表示的平均，来表示该类的特征.适用于nn.Parameter()
            # self.types_embedding.weight.data[type] = all_hiddens[ent_true_type_ids == type].mean(0) # 适用于nn.Embedding()，存在weight

    def get_types_embedding(self, ent_Ftype_ids):
        # ent_Ftype_ids的输入形式为(Seq_num, Max_ents_num, Task_type_num)
        return self.types_embedding[ent_Ftype_ids]






