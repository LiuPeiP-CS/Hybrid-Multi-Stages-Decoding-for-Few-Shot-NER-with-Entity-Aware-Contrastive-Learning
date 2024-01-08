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

logger = logging.getLogger(__file__) # 构建日志对象

class InputExample(object):
    def __init__(self, words, labels, types, entities):
        self.words = words # 每个任务中的每个seq words(一个句子), 没加[CLS]和[SEP]
        self.labels = labels # 每个任务中的每个seq words的实体检测标签BIO/BIOES/IO，没添加针对[CLS]和[SEP]的'O'。参考函数_convert_label_to_BIOES_的功能
        self.types = types # 每个任务中的实体类型
        self.entities = entities # 每个任务中的每个seq tokens构建的实体列表，每个实体包含了实体的始末位置(原始word的位置，不含CLS和SEP)和类型。参考函数_convert_label_entities_的功能


class InputFeature(object):
    def __init__(
        self,
        input_ids,
        input_mask,
        segment_ids,
        label_ids,
        ent_mask,
        ent_Ftype_ids,
        ent_type_mask,
        task_types_ids,
        entities,
    ):
        self.input_ids = input_ids # 前4个变量，对应bert的输入
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.ent_mask = ent_mask # max_ent_num * max_seq_len,每个实体在一个序列中进行mask
        self.ent_Ftype_ids = ent_Ftype_ids # max_ent_num * (1+negative_type_num), 每个元素为[ent_type_id, negative_type1, negative_type_id2, ... negative_num])
        self.ent_type_mask = ent_type_mask # shape = max_ent_num * (1+negative_type_num)，其中存在实体的前ent_num * (1+negative_type_num)为1，其余部分为0
        self.task_types_ids = task_types_ids # 该任务task中所有的实体类型id,
        self.entities = entities # 该序列中实体的列表[(ent_s, ent_e, ent_t), ...], ent_s和ent_e是在添加了['CLS']的前提下的始末位置，ent_t是实体类型的id


class Corpus(object):
    def __init__(self,
            logger,
            data_path,  # 数据集的来源地址
            bert_model,
            max_seq_len,
            label_list,  # 字符标签构造的列表。与类型list不一样，类型用于第二阶段，这个用于第一阶段
            entity_types,  # EntityTypes类型的对象。对象中的self.types_map是来自类型文件里的所有类型构建的
            do_lower_case=True,
            shuffle=True, # 是否扰乱task的获取顺序
            tagging="BIO", # 'OBI or OBIES'，默认BIO
            viterbi='none', # viterbi的类型，默认hard
            concat_types='None', # 如何将类向量进行表示，衔接进入
            dataset='FewNERD', # str类型，数据集名称, FewNERD/CrossDomain/CrossDomain2
            device='cuda',
            negative_types_number = -1 # 虚参，不是实参传入的内容。默认值就是-1
            ):
        self.logger = logger
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)
        self.max_seq_len = max_seq_len
        self.entity_types = entity_types
        self.label_list = label_list
        self.label2id = {label: id for id, label in enumerate(self.label_list)}
        self.id2label = {id: label for id, label in enumerate(self.label_list)}
        self.label_num = len(self.label_list)
        self.tagging_scheme = tagging
        self.max_len_EntTypSent = {'entity': 0, 'type': 0, 'sentence': 0}
        self.max_ent_num = 50 # entity span所包含的最多token数量
        self.viterbi = viterbi # viterbi的类型选择
        self.device = device
        self.dataset = dataset # str类型，数据集名称, FewNERD/CrossDomain/CrossDomain2
        self.negative_types_number = negative_types_number

        self.update_transition_matrix = False # 默认是False, 在测试时viterbi是hard，因此viterbi不需要调整；但在训练时，需要进行调整
        self.transition_matrix = torch.zeros([self.label_num, self.label_num], device=self.device) + 1e-9 # 初始化Matrix

        self.viterbi_construction() # 对viterbi中的矩阵内容进行填充

        self.tasks = self.read_tasks_from_file(data_path, self.update_transition_matrix, concat_types, dataset) # 各任务中，negative＿type＿num可能是不一样的。
        self.task_nums = len(self.tasks)

        self.batch_start_idx = 0
        self.batch_idxs = (
            np.random.permutation(self.task_nums) if shuffle else np.array([i for i in range(self.task_nums)])
        )

    def viterbi_construction(self):
        logger.info("Construct the transition matrix via {} scheme...".format(self.viterbi))

        # 其实就是构建Matrix[i][j]
        if self.viterbi == "none":
            self.transition_matrix = None
        elif self.viterbi == 'hard':
            if self.label_num == 3: # 实际上就是OBI
                self.transition_matrix[2][0] = -10000 # p(O—>I) = 0
            elif self.label_num == 5: # O-0, B-1, I-2, E-3, S-4
                for (i, j) in [
                    (2, 0), # p(O—>I) = 0
                    (3, 0), # p(O—>E) = 0
                    (0, 1), # p(B—>O) = 0
                    (1, 1), # p(B—>B) = 0
                    (4, 1), # p(B—>S) = 0
                    (0, 2), # p(I—>O) = 0
                    (1, 2), # p(I—>B) = 0
                    (4, 2), # p(I—>S) = 0
                    (2, 3), # p(E—>I) = 0
                    (3, 3), # p(E—>E) = 0
                    (2, 4), # p(S—>I) = 0
                    (3, 4), # p(S—>E) = 0
                ]:
                    self.transition_matrix[i][j] = -10000
            else:
                raise ValueError()
        elif self.viterbi == 'soft':
            self.update_transition_matrix = True

    def read_tasks_from_file(self, data_path, update_transition_matrix, concat_types, dataset):
        """
        :param data_path: 数据集的来源地址
        :param update_transition_matrix: 是否更新transition_matrix
        :param concat_types: 类型向量连接到序列文本的方式，past/before
        :param dataset: str类型数据，表示是哪个数据集
        :return: List[task1={'support': List[InputExample], 'query': List[InputExample]}, task2={}, ...]
        """
        self.logger.info('Reading tasks from {}...'.format(data_path))
        self.logger.info('Update_transition_matrix is {}'.format(update_transition_matrix))
        self.logger.info('Concat_types is {}'.format(concat_types))

        # *************** 读取获得所有任务 ****************
        with open(data_path, 'r', encoding='utf-8') as json_file:
            tasks_list = list(json_file) # tasks_list的每个元素都是一个任务，即原始数据集文件中的一行内容。即{'support': {'word': [[xxx], ...], 'label': [[xxx], ...]}, 'query': {'word': [[xxx], ...], 'label': [[xxx], ...]}, 'types':[xxx, ...]}
        all_span_labels = [] if update_transition_matrix else None # 默认为None。一个数据文件中，所有任务里用于第一阶段区分实体的BIO/BIOES标签列表集合
        if dataset == 'CrossDomain':
            tasks_list = self.convertDomain2FewNERD(tasks_list)

        # *************** 对每个任务的数据进行处理 ****************
        output_tasks = []
        for task_id, task_str in enumerate(tasks_list):
            # print(type(task))
            if task_id % 1000 == 0:
                self.logger.info('Reading tasks %d of %d', task_id, len(tasks_list))
            if dataset != 'CrossDomain':
                task = json.loads(task_str) # 是文件流中获取的字典形式的内容，需要转换一下
            else:
                task = task_str # 因为在Domain数据集中，task_str已经被处理成为了字典
            # 以上处理每个任务成为一个字典

            # *************** 对每个任务的类型信息进行处理 ****************
            types = task['types'] # 一个任务中的types部分

            if self.negative_types_number == -1: # 就是默认值，即-1
                self.negative_types_number = len(types) - 1 # 类型的负样本数量。第一个位置表示的是实际类型标签
            self.max_len_EntTypSent['type'] = max(self.max_len_EntTypSent['type'], len(types)) # 获取所有task中，type最多的数字

            # 对每个任务的类型描述，处理成为bert的输入token形式
            if concat_types != 'None': # past/before
                tokenized_types = self._tokenize_types_(types, concat_types) # bert tokenizer分解后的该task中的types。没有首尾CLS和SEP
            else:
                tokenized_types = None

            # *************** 对每个任务的support信息进行处理 ****************

            support = task['support']  # 一个任务中的support部分
            task_support_features, task_support_token_nums, task_support_span_labels = self.InfoGather4SupQue(support, types, tokenized_types, concat_types)
            # task_support_features: 一个task任务中，所有的seq的feature对象构造的列表：[seq1_feature, seq2_feature, ...]
            # task_support_token_nums: token_num的列表。token_num表示一个task任务的一个文本序列中，每个word经过tokenize后最后一个token在序列中的位置(在['CLS']的基础上)。
            if update_transition_matrix: # 训练数据的情况下
                all_span_labels.extend(task_support_span_labels)  # 原始数据中的数据标注

            # ************** 同理对每个任务的query信息进行处理 ***************
            query = task['query']
            task_query_features, task_query_token_nums, task_query_span_labels = self.InfoGather4SupQue(query, types, tokenized_types, concat_types)
            if update_transition_matrix:# 训练数据的情况下。此处的query应该是train中的
                all_span_labels.extend(task_query_span_labels)  # 原始数据中的数据标注

            output_tasks.append({
                'support_features': task_support_features,
                'support_token_nums': task_support_token_nums,
                'query_features': task_query_features,
                'query_token_nums': task_query_token_nums, # 该元素是列表的列表，[[seq1_cls, seq1_num0, seq1_num1, ...], ...seq_num].计算源自函数_convert_example_to_feature_
            })

        self.logger.info(
            "In %s: Max Entities Nums: %d, Max batch Types Number: %d, Max sentence Length: %d",
            data_path,
            self.max_len_EntTypSent['entity'], # (所有任务中)一个文本序列中，实体最多的数量
            self.max_len_EntTypSent['type'], # 该文件(的所有任务)中，实体类型最大的数量
            self.max_len_EntTypSent['sentence'], # 所有任务中，序列最长的数量(tokenize之后的最多token数量)
        )

        if update_transition_matrix:
            self._count_transition_matrix_(all_span_labels)
        return output_tasks

    def convertDomain2FewNERD(self, data):
        """
        :param data: 输入data是？？？需要下载数据去考察,是否文件名称的列表？
        :return: 将Domain的数据类型转换成为FewNERD的数据类型，进行模型的加载和使用
        """
        def decode_batch(batch: dict):
            word = batch['seq_ins']
            label = [
                [word_label.replace('B-', '').replace('I-', '') for word_label in seq_labels] for seq_labels in batch['seq_outs']
            ]
            return {'word': word, 'label': label}

        data = json.loads(data[0])
        res = []
        for domain in data.keys():
            d = data[domain] # value值
            labels = self.entity_types.types[domain] # 实体数据的类型标签
            res.extend([ # 列表中添加新的字典，字典的形式和FewNERD相同
                {'support': decode_batch(dataPiece['support']), 'query': decode_batch(dataPiece['batch']), 'types': labels} for dataPiece in d
            ])
        return res

    def _tokenize_types_(self, types, concat_types):
        """
        :param types: 每个任务上，所有类型标签所构建的列表
        :param concat_types: past/before，类型向量的连接方式
        :return:
        """
        types_tokens = []
        for t in types: # t是每个类型标签
            if 'embedding' in concat_types: # 实际上不存在，因为concat_type送past
                t_tokens = ['[unused{}]'.format(self.entity_types.types_map[t])]
            else:
                t_tokens = self.tokenizer.tokenize(t) # 将原始的str标签分解成为bert的需求模式
            if len(t_tokens) == 0:
                continue

            types_tokens.extend(t_tokens)
            types_tokens.append(',') # 使用','将不同类型区分开
        types_tokens.pop() # 删除最后一个','
        return types_tokens # 不包含['CLS']和['SEP']

    def InfoGather4SupQue(self, supque, types, tokenized_types, concat_types):
        """
        该函数用于处理一任务task.
        :param supque: 是support内容或者query内容
        :types: 该任务中的types
        :tokenized_types: task中所有的types经过tokenize之后的列表，由_tokenize_types_得到
        :concat_types: 类型(本task中的所有类型)描述与文本序列的拼接方式
        :return:
        """
        task_supque_features = []
        task_supque_token_nums = []
        task_span_labels = []

        for ith_sent, (words, labels) in enumerate(zip(supque['word'], supque['label'])):
            # 注意，这里的(words, labels)是一句话(token sequence)，以及相应的标签序列
            entities_SET = self._convert_label_to_entities_(labels)  # List[(s, e, type), ...]，指所有实体的始末位置和类型
            self.max_len_EntTypSent['entity'] = max(len(entities_SET),
                                                    self.max_len_EntTypSent['entity'])  # 获取所有task中，support中句子里实体最多的数量

            if self.tagging_scheme == 'BIEOS':
                span_labels = self._convert_label_to_BIEOS_(labels)
            elif self.tagging_scheme == 'BIO':
                span_labels = self._convert_label_to_BIO_(labels)
            elif self.tagging_scheme == 'IO':
                span_labels = self._convert_label_to_IO_(labels)
            else:
                raise ValueError('Invalid tagging scheme!')
            task_span_labels.append(span_labels) # 此处的span_labels，是原始序列标签的span标注，即用于entity-span的检测。如　O O O B I I O O

            # 将内容转换成为特征对象
            feature, token_sum = self._convert_example_to_feature_(
                InputExample(words, span_labels, types, entities_SET),
                tokenized_types=tokenized_types,  # 该任务中，tokenize之后的类型列表
                concat_types=concat_types  #
            ) # 处理的对象和结果是面向一个子序列的
            task_supque_features.append(feature)
            task_supque_token_nums.append(token_sum)

        return task_supque_features, task_supque_token_nums, task_span_labels # 结果是包含一个任务task中的各个句子序列的表示情况

    def _convert_label_to_entities_(self, labels):
        """
        :param labels: support数据中，一个token sequence所对应的标签序列
        :return: List[(s, e, type), ...]，指所有实体的始末位置和类型。s和e都是指的实际标注数据中的token的位置
        """
        tokens_num = len(labels) # seq中token的数量
        ent_ss = [
            ith_token
            for ith_token in range(tokens_num)
            if labels[ith_token] != 'O' and (not ith_token or labels[ith_token] != labels[ith_token-1])
        ] # 所有实体的开始字符的位置

        ent_es = [
            ith_token
            for ith_token in range(tokens_num)
            if labels[ith_token] != 'O' and (ith_token == tokens_num-1 or labels[ith_token] != labels[ith_token+1])
        ]
        return [(ent_s, ent_e, labels[ent_s]) for ent_s, ent_e in zip(ent_ss, ent_es)] # 此处的ent_e是实体的最后一个token的位置，如果需要获取ent_span，需要在ent_e+1

    def _convert_label_to_BIEOS_(self, labels):
        """
        :param labels: 原始句子中，每个词序列的标签列表
        :return: 原始序列标签的span标注，即用于entity-span的检测
        """
        res = []
        label_list = ['O'] + labels + ['O'] # 添加的前后两个，表示的是cls和sep的标签

        for i in range(1, len(label_list)-1): # 只考察中间的labels
            if label_list[i] == 'O':
                res.append('O')
                continue
            if label_list[i] != label_list[i-1] and label_list[i] != label_list[i+1]:
                res.append('S')
            elif label_list[i] != label_list[i-1] and label_list[i] == label_list[i+1]:
                res.append('B')
            elif label_list[i] == label_list[i-1] and label_list[i] != label_list[i+1]:
                res.append("E")
            elif label_list[i] == label_list[i-1] and label_list[i] == label_list[i+1]:
                res.append("I")
            else:
                raise ValueError("Some bugs exist in your code!")
        return res

    def _convert_label_to_BIO_(self, labels):
        """
        :param labels: 同上
        :return: 但是返回并没有加cls和sep的标签？
        """
        pre_ch = ''
        res = []
        for label in labels:
            if label == 'O':
                res.append('O')
            elif label != pre_ch:
                res.append('B')
            else:
                res.append('I')
            pre_ch = label
        return res

    def _convert_label_to_IO_(self, labels):
        """
        :param labels: 同上
        :return: 同上
        """
        res = []
        for label in labels:
            if label == "O":
                res.append("O")
            else:
                res.append("I")
        return res

    def _convert_example_to_feature_(
            self,
            example, # 主要由一个任务中的seq所构建的InputExample对象
            cls_token_at_end = False, # 默认为该值,面向BERT。如果是True，形式为A+[SEP]+B+[SEP]+[CLS]，面向XLNet/GPT
            cls_token = "[CLS]", # 默认为该值
            cls_token_segment_id = 0, # 默认为该值，BERT是0, XLNet是2
            sep_token = "[SEP]", # 默认为该值
            sep_token_extra = False, # 默认为该值,使用BERT模型；如果是True使用RoBERTa模型
            pad_on_left=False, # 默认为该值
            pad_token=0, # 默认为该值
            pad_token_segment_id=0, # 默认为该值
            pad_token_label_id=-1, # 默认为该值
            sequence_a_segment_id=0, # 默认为该值
            mask_padding_with_zero=True, # 默认为该值，输入为真实值的地方为1
            ignore_token_label_id=torch.nn.CrossEntropyLoss().ignore_index, # 默认为该值
            sequence_b_segment_id=1, # 默认为该值，表示bert的segment id
            tokenized_types=None, # 一个task任务中，tokenize之后的类型列表(不包含['CLS']和['SEP'])。参考函数_tokenize_types_的功能
            concat_types: str = "None",):
        """
        该函数面向一个task中的一个seq进行处理
        : example中的内容来自于InputExample类对象
        :return:
        """
        tokens, label_ids, token_sum = [], [], [1] # 这里的token_sum其实表示的是，word经过tokenize分析后，它的最后一个token在列表中的位置。而第一个1，实际上用于BERT输入前的['CLS]
        ent_num = len(example.entities)  # 本条数据中实体的数量

        if tokenized_types is None: # 只有在concat_type==None时才成立
            tokenized_types = []
        if 'before' in concat_types: # concat_type一般是past
            token_sum[-1] += 1 + len(tokenized_types) # token_sum的最后一个元素，表示的是task中types tokens的长度

        # 下列循环构建数据，用于BERT形式的训练测试
        for word, label in zip(example.words, example.labels):
            word_tokens = self.tokenizer.tokenize(word) # 对seq中的每个word进行tokenize处理
            token_sum.append(token_sum[-1] + len(word_tokens)) # 表示每个词段的长度，[tl, tl+wl1, tl+l1+l2, tl+l1+l2+l3, ...]
            if len(word_tokens) == 0:
                continue
            tokens.extend(word_tokens)
            label_ids.extend([self.label2id[label]] + [ignore_token_label_id] * (len(word_tokens)-1)) # 如果一个词分成多个token，第一个token保持原标签，其余token使用补充标签

        self.max_len_EntTypSent['sentence'] = max(self.max_len_EntTypSent['sentence'], len(tokens)) # 最长的句子长度

        """
        实体位置标记(4,6)　　　　　　　　　　 0      1     2      3         4       5        6         7       8       9
        原始标记(span-label)   　　　　     O      O     O      O         e       e        e         O       O       O
        token_num(假设的word分解)    1  (2,3,4) (5,6) (7,8) (9,10,11) (12,13) (14,15) (16,17,18) (19,20) (21,22) (23,24)
        token_num_iter            0(CLS)  1      2     3      4         5       6        7        8        9      10
        tokenize后实体范围　　　　　　　　　　　　　　　　　　　　　　11      ..............     17
        tokenize后seq的实际token信息(含CLS):
                                    0  (1,2,3) (4,5) (6,7) (8,9,10) (11,12) (13,14) (15,16,17) (18,19) (20,21) (22,23)
        """
        ent_se_token_ids = [(token_sum[ent_S], token_sum[ent_E+1]-1) for ent_S, ent_E, _ in example.entities] # 这里的实体所在的token的位置，在原始序列补充['CLS']的基础上
        ent_type_ids = [self.entity_types.types_map[ent_type] for _, _, ent_type in example.entities]  # [每个实体的类型id, ...]
        entities = [(ent_s, ent_e, ent_t) for (ent_s, ent_e), ent_t in zip(ent_se_token_ids, ent_type_ids)]  # ent_s和ent_e是在添加了['CLS']的前提下的位置，ent_t是类型的id

        ent_mask = np.zeros((self.max_ent_num, self.max_seq_len), np.int8) # 面向实体位置的mask
        for idx, (ent_startId, ent_endId) in enumerate(ent_se_token_ids):
            ent_mask[idx][ent_startId: ent_endId+1] = 1 # idx表示第几个实体。每个实体使用一个单独的seq序列进行mask标注

        ent_type_mask = np.zeros((self.max_ent_num, 1+self.negative_types_number), np.int8) # 面向实体类型的mask
        ent_type_mask[: ent_num, :] = np.ones((ent_num, 1+self.negative_types_number), np.int8)

        task_types_ids = [self.entity_types.types_map[_task_type] for _task_type in example.types] # 该任务中实体类型的id
        ent_Ftype_ids = np.zeros((self.max_ent_num, 1+self.negative_types_number), np.int8)
        ent_Ftype_ids[: ent_num, :] = self.entity_types.generate_negative_types(
            ent_type_ids, # 当前句子序列中的实际实体类型列表
            task_types_ids, # 当前句子所在的任务中，所有的types
            self.negative_types_number, # 样本的负类型数量, 实际上是K-1
        ) # shape = (ent_num, 1+negative_type_num), ent_num中的每个元素为[ent_type_id, negative_type1, negative_type_id2, ... negative_num])

        special_tokens_num = 3 if sep_token_extra else 2 # 3是面向RoBERTa
        available_seq_len = self.max_seq_len - special_tokens_num - len(tokenized_types) # 去除task_type信息、特殊添加符后，所能容忍的最大token长度
        if len(tokens) > available_seq_len:
            # 需要裁剪
            tokens = tokens[: available_seq_len]
            label_ids = label_ids[: available_seq_len]

        # ********** 实现bert输入前的数据填充和准备 **********
        orig_seq_len = len(tokens)
        if 'before' == concat_types:
            tokens = [cls_token] + tokenized_types + [sep_token] + tokens # tokens和label_ids是原始文本序列的id
            label_ids = [ignore_token_label_id] * (len(tokenized_types) + 2) + label_ids
            segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokenized_types) + 1) + [sequence_b_segment_id] * orig_seq_len

        else: ######### 其实就是past或者None, 将类型向量添加到文本序列的后面
            tokens = [cls_token] + tokens + [sep_token]
            label_ids = [ignore_token_label_id] + label_ids + [ignore_token_label_id]
            segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (orig_seq_len + 1)
            if "past" in concat_types:
                tokens += tokenized_types
                label_ids += [ignore_token_label_id] * len(tokenized_types)
                segment_ids += [sequence_b_segment_id] * len(tokenized_types)

        input_token_ids = self.tokenizer.convert_tokens_to_ids(tokens) # 原始的token转换成为bert输入的token_id
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_token_ids)
        # 数据填充完整
        padding_len = self.max_seq_len - len(input_token_ids)

        if pad_on_left: # 默认为False，补充信息到最长输入
            input_token_ids = [pad_token] * padding_len + input_token_ids
            input_mask = [0 if mask_padding_with_zero else 1] * padding_len + input_mask
            segment_ids =[pad_token_segment_id] * padding_len + segment_ids
            label_ids = [pad_token_label_id] * padding_len + label_ids
        else: # 这个是实际应用的参数，
            input_token_ids += [pad_token] * padding_len
            input_mask += [0 if mask_padding_with_zero else 1] * padding_len
            segment_ids +=[pad_token_segment_id] * padding_len
            label_ids += [pad_token_label_id] * padding_len

        assert len(input_token_ids) == self.max_seq_len
        assert len(input_mask) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len
        assert len(label_ids) == self.max_seq_len

        FO = InputFeature(
            input_ids = input_token_ids, # 原始序列经过tokenize, 并添加CLS和SEP后进行填充
            input_mask = input_mask,
            segment_ids = segment_ids,
            label_ids = label_ids, # 关于BIO的事实标签
            ent_mask = ent_mask, # max_ent_num * max_seq_len,每个实体在一个序列中进行mask
            ent_Ftype_ids = ent_Ftype_ids, # max_ent_num * (1+negative_type_num), 每个元素为[ent_type_id, negative_type1, negative_type_id2, ... negative_num])
            ent_type_mask = np.array(ent_type_mask), # shape = max_ent_num * (1+negative_type_num)，其中存在实体的前ent_num * (1+negative_type_num)为1，其余部分为0
            task_types_ids = np.array(task_types_ids), # 该任务task中所有的实体类型id,
            entities = entities, # 该序列中实体的列表[(ent_s, ent_e, ent_t), ...], ent_s和ent_e是在添加了['CLS']的前提下的始末位置，ent_t是实体类型的id
        )
        return FO, token_sum

    def _count_transition_matrix_(self, all_labels):
        """
        在训练阶段的函数使用...
        :param all_labels: 一个文件中，所有任务task中用于第一阶段区分实体的BIO/BIOES标签列表集合
        :return:
        """
        self.logger.info("Computing transition matrix...")
        for sent_labels in all_labels:
            for word_idx in range(len(sent_labels)-1):
                start = self.label2id[sent_labels[word_idx]]
                end = self.label2id[sent_labels[word_idx+1]]
                self.transition_matrix[end][start] += 1

        self.transition_matrix /= torch.sum(self.transition_matrix, dim=0)
        self.transition_matrix = torch.log(self.transition_matrix)
        self.logger.info('The Transition Matrix is Computed Done...')

    def decode_bpe_index(self, sent_spans):
        """
        :param sent_spans: 什么样的数据？？？需要解答
        :return:
        """
        res = []
        tokens = [sent_tokens for task in self.tasks for sent_tokens in task['query_token_nums']]
        assert len(tokens) == len(sent_spans), 'All token_nums: {}, all pred sent_nums: {}'.format(len(tokens), len(sent_spans))
        for sent_idx, spans in enumerate(sent_spans): # spans是来自sent_spans中的一个句子的表示
            sent_tokens = tokens[sent_idx] # sent_tokens是来自task['query_token_nums']中的一个句子的表示，起始位置多
            ent_se_in_seq = []
            for start, end in spans: # spans也是代表entity-span在序列中的始末位置，是token在seq中的位置，类似于token_num。
                ns = bisect.bisect_left(sent_tokens, start) # 返回start值在sent_tokens中的位置
                ne = bisect.bisect_left(sent_tokens, end)
                ent_se_in_seq.append((ns, ne)) # 实体所在的span中，头词和尾词的位置。注意，是原始的word位置，不是tokenized_token位置。从0开始(默认序列前面加上'[CLS]',也就是0)
            res.append(ent_se_in_seq)
        return res # 序列中实体的实际位置(词索引不是token索引，包含['CLS']，从0开始)

    def get_batch_meta(self, batch_size, device='cuda', shuffle=True):
        if self.batch_start_idx + batch_size > self.task_nums:
            self.reset_batch_info(shuffle=shuffle)
        query_batch, support_batch = [], []
        start_id = self.batch_start_idx

        for i in range(start_id, start_id + batch_size):
            # 获取每个任务及其索引
            task_id = self.batch_idxs[i]
            task_cur = self.tasks[task_id]

            # 获取每个任务的详情：每个任务(第185行，详情在第246行)里面的support或者query都是由多个句子构成的。每个句子(详情在第357行)是一个InputFeature对象。
            # 构建query的batch数据
            query_item = {
                "input_ids": torch.tensor(
                    [each_sent.input_ids for each_sent in task_cur["query_features"]], dtype=torch.long
                ).to(device),
                "input_mask": torch.tensor(
                    [each_sent.input_mask for each_sent in task_cur["query_features"]], dtype=torch.long
                ).to(device),
                "segment_ids": torch.tensor(
                    [each_sent.segment_ids for each_sent in task_cur["query_features"]], dtype=torch.long
                ).to(device),
                "label_ids": torch.tensor(
                    [each_sent.label_ids for each_sent in task_cur["query_features"]], dtype=torch.long
                ).to(device),
                "ent_mask": torch.tensor(
                    [each_sent.ent_mask for each_sent in task_cur["query_features"]], dtype=torch.int
                ).to(device),
                "ent_Ftype_ids": torch.tensor(
                    [each_sent.ent_Ftype_ids for each_sent in task_cur["query_features"]], dtype=torch.long
                ).to(device),
                "ent_type_mask": torch.tensor(
                    [each_sent.ent_type_mask for each_sent in task_cur["query_features"]], dtype=torch.int
                ).to(device),
                "task_types_ids": [each_sent.task_types_ids for each_sent in task_cur["query_features"]],
                "entities": [each_sent.entities for each_sent in task_cur["query_features"]],
                "task_id": task_id,
            }
            query_batch.append(query_item)


            # 构建support的batch数据
            support_item = {
                "input_ids": torch.tensor(
                    [each_sent.input_ids for each_sent in task_cur["support_features"]], dtype=torch.long
                ).to(device),
                "input_mask": torch.tensor(
                    [each_sent.input_mask for each_sent in task_cur["support_features"]], dtype=torch.long
                ).to(device),
                "segment_ids": torch.tensor(
                    [each_sent.segment_ids for each_sent in task_cur["support_features"]], dtype=torch.long
                ).to(device),
                "label_ids": torch.tensor(
                    [each_sent.label_ids for each_sent in task_cur["support_features"]], dtype=torch.long
                ).to(device),
                "ent_mask": torch.tensor(
                    [each_sent.ent_mask for each_sent in task_cur["support_features"]], dtype=torch.int
                ).to(device),
                "ent_Ftype_ids": torch.tensor(
                    [each_sent.ent_Ftype_ids for each_sent in task_cur["support_features"]], dtype=torch.long
                ).to(device),
                "ent_type_mask": torch.tensor(
                    [each_sent.ent_type_mask for each_sent in task_cur["support_features"]], dtype=torch.int
                ).to(device),
                "task_types_ids": [each_sent.task_types_ids for each_sent in task_cur["support_features"]],
                "entities": [each_sent.entities for each_sent in task_cur["support_features"]],
                "task_id": task_id,
            }
            support_batch.append(support_item)

        # 重新设置下一个batch的初始位置
        self.batch_start_idx += batch_size

        return query_batch, support_batch

    def reset_batch_info(self, shuffle=False):
        self.batch_start_idx = 0
        self.batch_idxs = (
            # 将真实的task_id进行打散混淆
            np.random.permutation(self.task_nums) if shuffle else np.array([i for i in range(self.task_nums)])
        )