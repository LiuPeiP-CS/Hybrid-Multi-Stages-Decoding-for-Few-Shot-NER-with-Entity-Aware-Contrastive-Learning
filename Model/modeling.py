#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/11/2 上午10:01
# @Author  : PeiP Liu
# @FileName: modeling.py
# @Software: PyCharm

import logging
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import BertForTokenClassification
from AugM.contner import ContrastiveLearning
from AugM.knnutils import *

logger = logging.getLogger(__file__)

class NBertForTokenClassification(BertForTokenClassification):
    def __init__(self, *args, **kwargs):
        super(NBertForTokenClassification, self).__init__(*args, **kwargs)
        self.input_size = 768
        self.span_loss_fct = CrossEntropyLoss(reduction='none') # 需要替换，未来使用CRF的检测损失
        self.type_loss_fct = CrossEntropyLoss(reduction='none') # 对实体类型的损失计算
        self.contral_loss = 0.0 # 聚类对比损失, 需要后期替换
        self.dropout = nn.Dropout(p=0.1)

        self.EntCont = ContrastiveLearning()

    def set_config(
            self,
            use_classify: bool = False,# 决定是否使用线性分类器进行标签分类
            distance_mode: str = 'cos', # 使用那种方式进行距离或者相似性度量，包括欧氏距离和cos
            similar_k: float = 30, # 使用最相邻的多少个邻居进行投票选择
            shared_bert: bool = True, # 是否共享bert模型
            train_mode: str = 'add', # 是哪种训练模式，两阶段模式还是共同模式
            ):
        self.use_classify = use_classify
        self.distance_mode = distance_mode
        self.similar_k = similar_k
        self.shared_bert = shared_bert
        self.train_mode = train_mode
        if train_mode == 'type':
            self.classifier = None

        if self.train_mode != 'span': # 两阶段的训练思路
            self.ln = nn.LayerNorm(768, 1e-5, True)
            if use_classify: # 第一阶段进行字符的分类，表示是否为实体的一部分
                self.type_classify = nn.Sequential(
                    nn.Linear(self.input_size, self.input_size*2),
                    nn.GELU(),
                    nn.Linear(self.input_size*2, self.input_size),
                )
# ****************************************************** 此处尝试改进KNN的思路 ****************************************************************
            if self.distance_mode != 'cos': # 第二阶段进行实体的分类，通过与周围的向量进行对比(典型的1-NN思路)
                self.dis_cls = nn.Sequential(
                    nn.Linear(self.input_size*3, self.input_size),
                    nn.GELU(),
                    nn.Linear(self.input_size, 2),
                )

        config = {
            "use_classify": use_classify,
            "distance_mode": distance_mode,
            "similar_k": similar_k,
            "shared_bert": shared_bert,
            "train_mode": train_mode,
        }

        logger.info("Model Setting: {}".format(config)) # 输出模型设置
        if not shared_bert: # 事实上，这个shared_bert是True
            self.bert2 = deepcopy(self.bert) # 这个bert的设置来自于继承的BertForTokenClassification

    def forward_ST(
            self,
            # 传入的参数变量都是列表的列表List[List[xxx]]，我们仅解释List[XXX]代表的含义
            input_ids, # 序列转换成为bert token后的token ids,该序列由原word seq串接了task中的types
            input_mask=None, # 面向input_ids的mask
            segment_ids=None, # 面向input_ids的0-1
            label_ids=None, # 面向input_ids的标签，标签是entity-span标签
            ent_mask=None,# max_ent_num * max_seq_len,每个实体在一个序列中进行mask
            ent_Ftype_ids=None, # max_ent_num * (1+negative_type_num), 每个元素为[ent_type_id, negative_type1, negative_type_id2, ... negative_num])
            ent_type_mask=None, # shape = max_ent_num * (1+negative_type_num)，其中存在实体的前ent_num * (1+negative_type_num)为1，其余部分为0
            entity_types=None, # 来自main函数中的EntityTypes类对象
            entity_mode: str = 'mean', # 获取entity-span向量的方式
            is_update_type_embedding: bool = False,
            lambda_max_loss: float = 0.0,
            sim_k: float = 0,
    ):
        # 对输入的原始token信息进行处理，便于输出bert编码特征
        # *********************************************************************#
        Max_seq_len = torch.nonzero((input_mask != 0).max(0)[0], as_tuple=False)[-1].item()+1 # 序列的最大长度
        input_ids = input_ids[:, :Max_seq_len] # 所有的数据进行长度对齐
        attention_mask = input_mask[:, :Max_seq_len].type(torch.int8)
        segment_ids = segment_ids[:, :Max_seq_len]
        labels = label_ids[:, :Max_seq_len] # 以最大实际信息为准，对填充信息进行截断

        output = self.bert(input_ids, attention_mask, segment_ids)
        seq_output = self.dropout(output[0]) # output[0]的shape是[batch_size, sequence_length, hidden_size]

        # **************************** 需要在此处进行CRF解码的改变 ******************************
        # 此处的token_span_logits是对token的分类，主要用于判断是否为entity-span的成分
        if self.train_mode != 'type':
            token_span_logits = self.classifier(seq_output) # 直接对token进行分类，(batch_size, seq_len, num_labels)
        else:
            token_span_logits = None # 实际上，我使用的是span模式下的分类，因此不一致

        if not self.shared_bert:
            output2 = self.bert2(input_ids, attention_mask, segment_ids)
            seq_output2 = self.dropout(output2[0])

        # *************************** 以下该部分，面向token的entity-span分类进行展开 ***************************
        if labels is not None and self.train_mode != 'type':
            # B, M, T = token_span_logits.shape
            Seq_num, Seq_len, _ = token_span_logits.shape
            if attention_mask is not None:
                active_atten_mask = attention_mask.view(-1) == 1 # (batch_size, seq_len)
                active_token_span_logits = token_span_logits.reshape(-1, self.num_labels)[
                    active_atten_mask]  # 相应位置上token关于ent-span标签的概率预测，(所有序列上有效的token集合数量，BIO类型的标签数量)
                active_true_span_labels = labels.reshape(-1)[active_atten_mask]  # 相应位置上token关于ent-span的实际标签，(所有序列上有效的token集合数量，)
                base_token_span_loss = self.span_loss_fct(active_token_span_logits, active_true_span_labels)  # 计算token的span分类损失,(所有序列上有效的token集合数量，)
                token_span_loss = torch.mean(base_token_span_loss)

                if lambda_max_loss > 0: # 附加句子中token的最大损失作为额外损失计算
                    active_atten_mask = active_atten_mask.view(Seq_num, Seq_len)
                    active_max = []
                    start_id = 0
                    for i in range(Seq_num):
                        sent_len = torch.sum(active_atten_mask[i])  # 在对元素为True和False的tensor进行求和时，True被表示为1，False被表示为0
                        end_id = start_id + sent_len
                        # base_token_span_loss[start_id: end_id]表示的是一个文本序列里面，每个token的损失。这表示，用序列里最大的token损失，来表示整个序列。
                        active_max.append(torch.max(base_token_span_loss[start_id: end_id]))
                        start_id = end_id

                    token_span_loss += lambda_max_loss * torch.mean(torch.stack(active_max)) # 对所有seq中(损失最大的token)对应的损失，按序列数量求平均
            else:
                raise ValueError('Miss attention mask!')

        else:
            token_span_loss = None

        # ********************************* 以下需要对entity做处理 ********************************* #
        if ent_Ftype_ids is not None and self.train_mode != 'span': # ent_Ftype_ids不为None，实际上指的是在Meta-training阶段和Meta-testing的finetuning阶段
            if ent_type_mask.sum() != 0: # 当ent_type_mask是面向实际的数据存在时
                # 获取该任务task中最大的实体数量
                Max_ents_num = torch.nonzero((ent_type_mask[:, :, 0] != 0).max(0)[0], as_tuple=False)[-1].item()+1
            else:
                Max_ents_num = 1
            ent_mask = ent_mask[:, :Max_ents_num, :Max_seq_len].type(torch.int8)
            ent_Ftype_ids = ent_Ftype_ids[:, :Max_ents_num, :] # 此处需要确定ent_Ftype_ids等数据的含义
            ent_type_mask = ent_type_mask[:, :Max_ents_num, :].type(torch.int8)

            # 获取实体的表示，即entity span embedding
            ent_out = self.get_ent_hidden(seq_output if self.shared_bert else seq_output2, ent_mask, entity_mode)
            if self.use_classify:
                ent_out = self.type_classify(ent_out) # 对特征进行线性变化
            ent_out = self.ln(ent_out) # 对特征进行layernorm, # (batch_size, max_ent, hidden_dim),这里的batch_size可以理解成为序列数量Seq_num

            if is_update_type_embedding:
                # 该步骤只应该出现在train_support, train_eval, test_support的数据集上
                entity_types.update_type_embedding(ent_out, ent_Ftype_ids, ent_type_mask)

            # ****************************************方法的改进应该从此处进行和开始**********************************************#

            Seq_num, Max_ents_num, Task_type_num = ent_Ftype_ids.shape  # 分别表示一个任务中的文本序列的数量，该任务的各序列中最大的实体数量，该任务中的type的数量
            ##### predictions. 在第二维度进行扩充。该变量表示的是实体的表示，可以认为是前馈神经网络的预测输出
            ent_out = ent_out.unsqueeze(2).expand(Seq_num, Max_ents_num, Task_type_num, -1)

            ##### ground-truth. (batch_size, max_ents, task_type_num, hidden_dims)。该变量表示的是，实体所属类型的特征表示。可以认为是实体的ground-truth type embedding
            ent_types_out = self.get_ent_types_embedding(ent_Ftype_ids, entity_types)

            # 以下的计算，是predictions和ground truth的相关性计算，也就是每个实体预测到task中各种类型的打分
            if self.distance_mode == 'cat':
                ent_types_score = torch.cat([ent_out, ent_types_out, abs(ent_out-ent_types_out)], -1)
                ent_types_score = self.dis_cls(ent_types_score) # (batch_size, max_ents, task_type_num, 1)
                ent_types_score = ent_types_score.squeeze(-1) # (batch_size, max_ents, task_type_num)
            elif self.distance_mode == 'l2':
                ent_types_score = -(torch.pow(ent_out - ent_types_out, 2)).sum(-1) # (batch_size, max_ents, task_type_num),表示在每个类别上的预测得分
            elif self.distance_mode == 'cos':
                sim_k = sim_k if sim_k else self.similar_k
                ent_types_score = sim_k * (ent_out * ent_types_out).sum(-1) / 768 # (batch_size, max_ents, task_type_num)
            ent_type_logits = ent_types_score # 对实体的类型预测，每个实体预测到task中各种类型的打分

            # 通过上述pred和true的相关性得分，来与0计算交叉熵损失
            if Max_ents_num:
                ent_type_maskC = ent_type_mask.clone()
                ent_type_maskC[ent_type_maskC.sum(-1)==0] = 1
                ent_type_pred_score = ent_types_score * ent_type_maskC # ent_type_pred_score与ent_types_score有什么不同？
                ent_type_label = torch.zeros((Seq_num, Max_ents_num)).to(input_ids.device)
                # 是否因为前面计算了pred和true的相关性，所以此处直接使用了全0进行计算？
                ent_type_loss = self.calc_loss(self.type_loss_fct, ent_type_pred_score, ent_type_label, ent_type_mask[:, :, 0])
            else:
                ent_type_loss = torch.tensor(0).to(seq_output.device)

        else:
            ent_type_logits, ent_type_loss = None, None

        return token_span_logits, ent_type_logits, token_span_loss, ent_type_loss
        # 返回值的尺寸以及含义：
        # token_span_logits是判断每个token为span形式标签(如BIO)的概率，shape = (batch_size, seq_len, BIO_num_labels)
        # ent_type_logits是判断一个实体为task中实体类型的概率(实际上是该实体的向量表示与所有task_types向量表示的相关性计算)，shape = (batch_size, max_ents, task_type_num)
        # token_span_loss表示token预测成为entity-span的损失
        # ent_type_loss表示entity预测成为task_type的损失

    def get_ent_hidden(self, hidden, ent_mask, ent_mode):
        """
        :param hidden: 输出的序列特征
        :param ent_mask: 每个实体在一个序列中进行mask标注
        :param ent_mode: 通过token embedding对实体的span embedding进行获取
        :return:
        """
        B, M, T = ent_mask.shape # 同(batch_size, max_ents, task_type_num)
        e_out = hidden.unsqueeze(1).expand(B, M, T, -1) * ent_mask.unsqueeze(
            -1)  # (batch_size, max_ents, seq_len, dim)
        if ent_mode == 'mean':
            return e_out.sum(2) / (ent_mask.sum(-1).unsqueeze(-1) + 1e-30)  # (batch_size, max_ent, hidden_dim)

        # 下面的两种方式需要改进填写
        # *********************************************************************#
        elif ent_mode == 'pool':
            return None
        elif ent_mode == 'attention':
            return None

    def get_ent_types_embedding(self, ent_Ftype_ids, ent_types):
        return ent_types.get_types_embedding(ent_Ftype_ids)

    def calc_loss(self, loss_fn, ent_type_pred_score, ent_type_label, mask=None):
        """
        对应实参：self.type_loss_fct, ent_type_pred_score, ent_type_label, ent_type_mask[:, :, 0]
        :param loss_fn: 损失计算函数
        :param ent_type_pred_score: (batch_size, max_ents, task_type_num), 每个实体预测到task中各种类型的打分
        :param ent_type_label: (Seq_num, Max_ents_num), 全0矩阵
        :param mask: (Seq_num, Max_ents_num)，在Max_ents_num序列中有实体的地方标为1
        :return:
        """
        ent_type_label = ent_type_label.reshape(-1)
        ent_type_pred_score += 1e-10
        ent_type_pred_score = ent_type_pred_score.reshape(-1, ent_type_pred_score.shape[-1])
        ce_loss = loss_fn(ent_type_pred_score, ent_type_label.long())
        if mask is not None:
            mask = mask.reshape(-1)
            ce_loss = ce_loss * mask
            return ce_loss.sum() / (mask.sum() + 1e-10)
        return ce_loss.sum() / (ent_type_label.sum() + 1e-10)

    def get_ent_embAlabel(self, task_data):
        input_ids = task_data["input_ids"],  # 序列转换成为bert token后的token ids,该序列由原word seq串接了task中的types
        input_mask = task_data["input_mask"],  # 面向input_ids的mask
        segment_ids = task_data["segment_ids"],  # 面向input_ids的0-1
        ent_mask = task_data["ent_mask"],  # max_ent_num * max_seq_len,每个实体在一个序列中进行mask
        ent_Ftype_ids = task_data["ent_Ftype_ids"],  # max_ent_num * (1+negative_type_num), 每个元素为[ent_type_id, negative_type1, negative_type_id2, ... negative_num])
        ent_type_mask = task_data["ent_type_mask"],  # shape = max_ent_num * (1+negative_type_num)，其中存在实体的前ent_num * (1+negative_type_num)为1，其余部分为0


        Max_seq_len = torch.nonzero((input_mask != 0).max(0)[0], as_tuple=False)[-1].item()+1 # 序列的最大长度
        input_ids = input_ids[:, :Max_seq_len] # 所有的数据进行长度对齐
        attention_mask = input_mask[:, :Max_seq_len].type(torch.int8)
        segment_ids = segment_ids[:, :Max_seq_len]

        output = self.bert(input_ids, attention_mask, segment_ids)
        seq_output = self.dropout(output[0])

        if not self.shared_bert:
            output2 = self.bert2(input_ids, attention_mask, segment_ids)
            seq_output2 = self.dropout(output2[0])

        if ent_type_mask.sum() != 0: # 当ent_type_mask是面向实际的数据存在时
            # 获取该任务task中最大的实体数量
            Max_ents_num = torch.nonzero((ent_type_mask[:, :, 0] != 0).max(0)[0], as_tuple=False)[-1].item()+1
        else:
            Max_ents_num = 1
        ent_mask = ent_mask[:, :Max_ents_num, :Max_seq_len].type(torch.int8)
        ent_Ftype_ids = ent_Ftype_ids[:, :Max_ents_num, :] # 此处需要确定ent_Ftype_ids等数据的含义
        ent_type_mask = ent_type_mask[:, :Max_ents_num, :].type(torch.int8)
        ent_out = self.get_ent_hidden(seq_output if self.shared_bert else seq_output2, ent_mask)
        if self.use_classify:
            ent_out = self.type_classify(ent_out)  # 对特征进行线性变化
        ent_out = self.ln(ent_out)

        return (ent_out, ent_Ftype_ids[:, :, 0], ent_type_mask[:, :, 0])


    def EntContr_loss(self, task_data, loss_type):
        ent_contra_loss = None
        if task_data['ent_Ftype_ids'] is not None and self.train_mode != 'span':  # ent_Ftype_ids不为None，实际上指的是在Meta-training阶段和Meta-testing的finetuning阶段
            ent_emb, ent_label, ent_mask = self.get_ent_embAlabel(task_data)
            ent_contra_loss = self.EntCont(ent_emb, ent_label, ent_mask, loss_type)

        return ent_contra_loss

    def preBuildDataStore(self, sup_task_data):
        DataStore = None
        if sup_task_data['ent_Ftype_ids'] is not None and self.train_mode != 'span':  # ent_Ftype_ids不为None，实际上指的是在Meta-training阶段和Meta-testing的finetuning阶段
            pad_ent_emb, pad_ent_labels, pad_ent_label_mask = self.get_ent_embAlabel(sup_task_data)
            # 按照KNN的思路进行datastore的设计(在KNNutils中构建函数)
            DataStore = build_datastore(pad_ent_emb, pad_ent_labels, pad_ent_label_mask)
        return DataStore

    def TestEntEmb(self, test_task_data):
        # 用于获取测试时，实体的向量嵌入
        ents_emb = None
        if test_task_data['ent_Ftype_ids'] is not None and self.train_mode != 'span':
            ents_emb, _, _ = self.get_ent_embAlabel(test_task_data) # 接下来需要看下forward_ST产生的ent_logits的shape，以及KNNNER需要的ent_logits的shape
        return ents_emb

    def FusingKNN(self, sup_Datastore, test_task_data, model_logits, args):
        # 输入包括监督数据的datastore和测试实体的向量嵌入，以及模型获取的实体分类的logits。返回值是KNN获取的logtis和模型获取的logits的结合(在KNNutils中构建函数)
        # 模型获得的logits的shape是(batch_size, max_ents, task_type_num)，需要用mask找到真实值
        ents_emb = self.TestEntEmb(test_task_data)
        return ModelAKNN4prob(sup_Datastore, ents_emb, model_logits, args)
