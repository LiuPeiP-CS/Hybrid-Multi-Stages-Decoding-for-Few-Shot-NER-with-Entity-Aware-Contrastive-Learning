#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/11/2 上午9:59
# @Author  : PeiP Liu
# @FileName: leaner.py
# @Software: PyCharm

import json
import logging
import os
import shutil
import time
from copy import deepcopy

import numpy as np
import torch
from torch import nn

import joblib
from Model.modeling import NBertForTokenClassification
from transformers import CONFIG_NAME, PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME
from transformers import AdamW as BertAdam
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__file__)

class Learner(nn.Module):
    ignore_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    pad_token_label_id = -1

    def __init__(
            self,
            bert_model, # bert模型的选择，str数据类型
            label_list, # 用于span的标签BIO/BIOES
            freeze_layer, # 固定bert的几层不用进行finetune
            logger, # 日志对象
            lr_meta, # 此处的学习率用于常规的监督训练，其实也是meta训练的一部分
            lr_inner, # 用于inner-update中的参数优化
            warmup_prop_meta, # ???warmup的用处？
            warmup_prop_inner, # ???warmup的用处？
            max_meta_steps, # 主要是搞清楚meta步骤和inner的区别
            model_dir='', # 用于存储训练好的模型的文件夹地址，当它不为空时，表示已经有训练好的模型。因此，不为空主要是为了进行测试和验证。
            cache_dir='', # bert的缓存地址, ''其实表示None
            gpu_no=0, # 默认为该值
            py_alias='python', # 默认为该值
            args=None,
    ):
        super(Learner, self).__init__()
        self.lr_meta = lr_meta
        self.lr_inner = lr_inner
        self.warmup_prop_meta = warmup_prop_meta
        self.warmup_prop_inner = warmup_prop_inner
        self.max_meta_steps = max_meta_steps

        self.bert_model = bert_model
        self.label_list = label_list
        self.py_alias = py_alias
        self.ent_types = args.ent_types # EntityType对象

        self.is_debug = args.debug # 是否为True是可选项，为了保存训练好的模型
        self.train_mode = args.train_mode
        self.eval_mode = args.eval_mode
        self.model_dir = model_dir
        self.args = args
        self.freeze_layer = freeze_layer
        self.loss_type = "euclidean" if args.K == 1 else "KL"

        num_labels = len(label_list) # BIO类型的标签数量

        # **************************************　在不同阶段的参数设置，此处需要进行考虑　************************************
        # 实际上，在self.eval_mode != 'two-stage'的情况下，load(saved_model)和新finetune相差性不大。无非就是load的内容是训练好的模型还是原始bert模型
        # 执行测试任务
        if model_dir != '':
            if self.eval_mode != 'two-stage': # 分为两阶段two-stage评估和同时add评估
                self.load_model(self.eval_mode)
        # 执行训练任务
        else:
            logger.info("********** Loading pre-trained model ***********")
            cache_dir = cache_dir if cache_dir else str(PYTORCH_PRETRAINED_BERT_CACHE)
            self.model = NBertForTokenClassification.from_pretrained(
                bert_model,
                cache_dir=cache_dir,
                num_labels=num_labels,
                output_hidden_states=True,
            ).to(args.device) # 初始化模型对象

        if self.eval_mode != 'two-stage': # 不是two-stage，那就是add
            self.model.set_config( # 对模型的内部变量进行设置
                args.use_classify,
                args.distance_mode,
                args.similar_k,
                args.shared_bert,
                self.train_mode,
            )
            self.model.to(args.device)
            self.layer_set() # 设置了优化器以及计划优化的神经网络层

    def load_model(self, mode: str = 'all'):
        """
        本函数的目的在于进行模型的加载，并设置模型里面的部分参数(实际上用处不大)
        :param mode: 模型训练时的模式，并使用该模式进行了模型的保存
        :return:
        """
        if not self.model_dir:
            return
        model_dir = self.model_dir
        logger.info("********** Loading saved {} model **********".format(mode))
        output_model_file = os.path.join(model_dir, "en_{}_{}".format(mode, WEIGHTS_NAME)) # 实际训练好的模型的参数

        # 以下对空模型结构进行某些设置
        self.model = NBertForTokenClassification.from_pretrained(
            self.bert_model,
            num_labels=len(self.label_list), # 用于span检测的BIO标签
            output_hidden_states=True,
        )
        self.model.set_config( # 在
            self.args.use_classify, # 决定是否使用线性分类器进行标签分类
            self.args.distance_mode, # 使用那种方式进行距离或者相似性度量，包括欧氏距离和cos
            self.args.similar_k, # 使用最相邻的多少个邻居进行投票选择
            self.args.shared_bert, # 是否共享bert模型
            mode, # 是哪种训练模式，两阶段模式还是共同模式
        )
        self.model.to(self.args.device)

        # 使用训练好的模型对空模型结构进行填充
        self.model.load_state_dict(torch.load(output_model_file, map_location='cuda'))

        # 加载后的模型的层信息进行设置
        self.layer_set()

    def layer_set(self):
        # 设置不进行finetune的参数
        no_grad_param_names = ['embeddings', 'pooler'] + [
            'layer.{}'.format(i) for i in range(self.freeze_layer)
        ]
        logger.info('The frozen parameters are: ')
        for name, param in self.model.named_parameters():
            if any(no_grad_pn in name for no_grad_pn in no_grad_param_names):
                param.requires_grad = False
                logger.info(" {}".format(name))


        # 对模型进行微调的优化器设置
        self.opt = BertAdam(self.get_optimizer_grouped_parameters(), lr=self.lr_meta) # 设置优化器
        self.scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=int(self.max_meta_steps*self.warmup_prop_meta), # 学习的初始化步数
            num_training_steps=self.max_meta_steps,
        )

    def get_optimizer_grouped_parameters(self):
        # 构建需要进行finetune(优化)的参数
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [ # 需要去优化的参数
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                'weight_decay': 0.01,
            },
            {
                'params': [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                'weight_decay': 0.0,
            }
        ]
        return optimizer_grouped_parameters

    def evaluate_meta_(
            self,
            corpus,
            logger,
            lr, # 用于在test过程中，对test-support的finetune
            steps, # max_ft_steps
            mode, # 选择intra或者inner
            set_type, # "valid"或者"test"
            type_steps: int = None, # max_type_ft_steps
            viterbi_decoder=None, # 默认是不会为None的，因为存在实际的viterbi
    ):
        if not type_steps: # 0的话，就和steps一致
            type_steps = steps
        if self.is_debug: # 默认False
            self.save_model(self.args.result_dir, 'begin', self.args.max_seq_len, 'all')

        # # 查看GPU的使用情况
        # current_gpu_index = torch.cuda.current_device()
        # used_memory = torch.cuda.memory_allocated(current_gpu_index) / (1024 ** 2)
        # logger.info("第一阶段测试前，已使用的GPU显存：{:.2f} MB".format(used_memory))

        logger.info('**************** Begin First Stage. ****************')
        if self.eval_mode == 'two-stage':
            self.load_model('span')
        # 实现针对span任务的权值传递
        names = self.get_names()
        params = self.get_params()
        weights = deepcopy(params)

        eval_loss = 0.0
        nb_eval_steps = 0
        # preds = None
        t_tmp = time.time()

        ents_true_bet, pred_task_ents_info, ents_pred_span_be, tasks_span_logits, tasks_type_preds, tasks_type_trues = [], [], [], [], [], []
        for item_id in range(corpus.task_nums):
            eval_batch_query, eval_batch_support = corpus.get_batch_meta(
                batch_size=1, shuffle=False, # 也就是，实际上是一个任务task内的数据
            ) # 数据的来源是由corpus决定的

            # finetuning on the eval support dataset (Meta-testing phase，包括了span和type)
            if not self.args.nouse_inner_ft: # 此步骤，理论上可有可无。但应该测试。
                self.inner_update(eval_batch_support[0], lr_curr=lr, inner_steps=steps) # batch_size大小为1，因此eval_batch_support[0]就是整个batch的数据

            # ××××××××××××××××××××　在此处构建面向valid中support的{emb: label}数据字典，用于选择最优的分类模型。如果选择模型时不用KNN，则应该删掉 ××××××××××××××××××××
            if self.args.use_cbknn and self.args.use_knn and self.args.K>1:
                supp4datastore1 = self.model.preBuildDataStore(eval_batch_support[0])

            # eval on pseudo query examples (test example)
            self.model.eval()
            with torch.no_grad():
                eval_task1 = { # 该batch指的是对一个task中的数据进行batch处理，与task无关
                        "input_ids": eval_batch_query[0]["input_ids"], # 同理，由于batch_size=1,　[0]表示一个任务的数据。
                        "input_mask": eval_batch_query[0]["input_mask"],
                        "segment_ids": eval_batch_query[0]["segment_ids"],
                        "label_ids": eval_batch_query[0]["label_ids"],
                        "ent_mask": eval_batch_query[0]["ent_mask"],
                        "ent_Ftype_ids": eval_batch_query[0]["ent_Ftype_ids"],
                        "ent_type_mask": eval_batch_query[0]["ent_type_mask"],
                        "entity_types": self.ent_types,
                }
                # 先对验证集算损失，再对其进行预测输出
                # 对事实数据的预测，即对每个token的span预测，对实际span的类型预测
                test_token_span_logits, test_ent_type_logits, test_token_span_loss, _ = self.batch_test(eval_task1)
                tasks_span_logits.append(test_token_span_logits)
                if self.model.train_mode != 'type':
                    eval_loss += test_token_span_loss # 可以认为面向的是span的损失
                if self.model.train_mode != 'span':

                    # ×××××××××××　在此处使用KNN的方法，更新　test_ent_type_logits　×××××××××××
                    if self.args.use_cbknn and self.args.use_knn and self.args.K>1:
                        test_ent_type_logits = self.model.FusingKNN(supp4datastore1, eval_task1, test_ent_type_logits, self.args)

                    # type_pred, type_ground
                    ent_type_pred, ent_type_true = self.eval_typing(
                        test_ent_type_logits, eval_batch_query[0]["ent_type_mask"]
                    ) # ent_type_pred的shape=(seq_num, max_ents), 根据事实span的type预测
                    tasks_type_preds.append(ent_type_pred)
                    tasks_type_trues.append(ent_type_true)
                else:
                    # ents_be_mask = (seq_num, max_ent, seq_len)
                    # ents_type_ids = (seq_num, max_ent, task_type_num), 每个ent对应该task中的所有entity type
                    # ents_type_mask = (seq_num, max_ent, task_type_num)。每个句子所对应的矩阵(max_ent, task_type_num)中，[:ent_num, :type_num]为1
                    # ents_pred_be, entity-span的预测位置，list(list(实体位置的元组)), 是预测的结果
                    # task_type_ids, list(list(该task中所有的type的id))
                    ents_be_mask, ents_type_ids, ents_type_mask, ents_pred_be, task_type_ids = self.decode_span(
                        test_token_span_logits,
                        eval_batch_query[0]["label_ids"],
                        eval_batch_query[0]["task_types_ids"],
                        eval_batch_query[0]["input_mask"],
                        viterbi_decoder,
                    )
                    ents_true_bet.extend(eval_batch_query[0]['entities']) # ents_true_bet: 列表的列表，list(list((ent_s, ent_e, ent_t))), 外部的list是一个task里的seq_num,里面的list是ent_num
                    ents_pred_span_be.extend(ents_pred_be) # ents_pred_span_be: 列表的列表，list(list(预测的实体位置的元组))

            nb_eval_steps += 1

            self.load_weights(names, weights)

            if item_id % 200 == 0:
                logger.info(
                    "  To sentence {}/{}. Time: {}sec".format(
                        item_id, corpus.task_nums, time.time() - t_tmp
                    )
                )


        # # 查看GPU的使用情况
        # current_gpu_index = torch.cuda.current_device()
        # used_memory = torch.cuda.memory_allocated(current_gpu_index) / (1024 ** 2)
        # logger.info("第二阶段测试前，已使用的GPU显存：{:.2f} MB".format(used_memory))


        logger.info('**************** Begin Second Stage. ****************')
        if self.eval_mode == "two-stage":
            self.load_model("type")
            names = self.get_names()
            params = self.get_params()
            weights = deepcopy(params)

        if self.train_mode == 'add':
            for task_id in range(corpus.task_nums):
                eval_batch_query, eval_batch_support = corpus.get_batch_meta(
                    batch_size=1, shuffle = False
                )
                # finetune train on eval support examples
                self.inner_update(eval_batch_support[0], lr_curr=lr, inner_steps=type_steps)

                # ××××××××××××××××××××　在此处构建面向test中support的{emb: label}数据字典，用于最终的预测输出 ××××××××××××××××××××
                if self.args.use_knn and self.args.K>1:
                    supp4datastore2 = self.model.preBuildDataStore(eval_batch_support[0])

                test_token_span_logits = tasks_span_logits[task_id]  # 获取出相应task的token_span_logits
                # eval on eval query examples (test example)
                self.model.eval()
                with torch.no_grad():
                    ents_be_mask, ents_type_ids, ents_type_mask, ents_pred_be, task_type_ids = self.decode_span(
                        test_token_span_logits,
                        eval_batch_query[0]["label_ids"],
                        eval_batch_query[0]["task_types_ids"],
                        eval_batch_query[0]["input_mask"],
                        viterbi_decoder,
                    ) # 对实体跨度的解析

                    eval_task2 = {  # 此处的ents_type_logits，是对预测entity-span的计算，源自于ents_be_mask
                        "input_ids": eval_batch_query[0]["input_ids"],
                        "input_mask": eval_batch_query[0]["input_mask"],
                        "segment_ids": eval_batch_query[0]["segment_ids"],
                        "label_ids": eval_batch_query[0]["label_ids"],
                        "ent_mask": ents_be_mask,  # (batch_size, max_ent, seq_len), 对每个实体，在序列中构造出他的mask
                        "ent_Ftype_ids": ents_type_ids,  # 每一个元素是一个矩阵，每个矩阵包含max_ent行，每一行是该条数据的全部类型id
                        "ent_type_mask": ents_type_mask,  # (batch_size, max_ent, type_num)全1的张量
                        "entity_types": self.ent_types,  # list(每条数据的实体类型)
                    }

                    _, ents_type_logits, _, ents_type_loss = self.batch_test(eval_task2) # 对实体类型的解析,来自于对预测span的解析

                    # ×××××××××××　在此处使用KNN的方法，更新　ents_type_logits　×××××××××××
                    if self.args.use_knn and self.args.K>1:
                        ents_type_logits = self.model.FusingKNN(supp4datastore2, eval_task2, ents_type_logits, self.args)

                    _, pred_seq_ents_info = self.decode_entity(
                        ents_type_logits, ents_pred_be, task_type_ids, eval_batch_query[0]["entities"]
                    )  # pred_seq_ents_info = List[List[元素]]，每个元素是(实体预测的始末位置，预测的实体类型，实体的预测分类打分)
                    pred_task_ents_info.extend(pred_seq_ents_info)

                    eval_loss += ents_type_loss

                    if self.eval_mode == "two-stage":
                        eval_task3 = {
                                "input_ids": eval_batch_query[0]["input_ids"],
                                "input_mask": eval_batch_query[0]["input_mask"],
                                "segment_ids": eval_batch_query[0]["segment_ids"],
                                "label_ids": eval_batch_query[0]["label_ids"],
                                "ent_mask": eval_batch_query[0]["ent_mask"],
                                "ent_Ftype_ids": eval_batch_query[0]["ent_Ftype_ids"],
                                "ent_type_mask": eval_batch_query[0]["ent_type_mask"],
                                "entity_types": self.ent_types,
                        }
                        test_token_span_logits, test_ent_type_logits, _, _ = self.batch_test(eval_task3)

                        # ×××××××××××　在此处使用KNN-NER的方法，更新　test_ent_type_logits　×××××××××××
                        if self.args.use_knn and self.args.K>1:
                            test_ent_type_logits = self.model.FusingKNN(supp4datastore2, eval_task3, test_ent_type_logits, self.args)

                        type_pred, type_ground = self.eval_typing(
                            test_ent_type_logits, eval_batch_query[0]["ent_type_mask"]
                        )  # type_pred是预测的实体类别。该类型是在task中type的顺序id，还需要转换成为具体的类型

                        tasks_type_preds.append(type_pred)
                        tasks_type_trues.append(type_ground)


                self.load_weights(names, weights)
                if task_id % 200 == 0:
                    logger.info(
                        "  To sentence {}/{}. Time: {}sec".format(
                            task_id, corpus.task_nums, time.time() - t_tmp
                        )
                    )
        average_eval_loss = eval_loss / nb_eval_steps

        #****************************************** 数据结果的保存 ************************************
        if self.is_debug:
            joblib.dump([ents_true_bet, pred_task_ents_info, ents_pred_span_be], "debug/TruePred_ents_info.pkl")
        store_dir = self.args.model_dir if self.args.model_dir else self.args.result_dir

        joblib.dump(
            [ents_true_bet, pred_task_ents_info, ents_pred_span_be],
            # pred_task_ents_info大小是(batch_size, ent_num)，ent_num的每个元素包含了实体的[(预测始末位置，预测实体类型，预测实体的分类打分),...]
            # ents_pred_span_be的每个元素是list(预测实体位置的元素), 是预测的结果
            "{}/{}_{}_preds.pkl".format(store_dir, "all", set_type),
        )

        joblib.dump(
            [tasks_type_trues, tasks_type_preds], # type_preds是(data_num, max_ent)，表示预测的实体类型
            "{}/{}_{}_preds.pkl".format(store_dir, "typing", set_type),
        )

        # ***************************** 性能评估 *****************************
        ents_pred_bet = [[ent_inf[:-1] for ent_inf in seq_ents_inf] for seq_ents_inf in pred_task_ents_info] # List[List[元素]]，实体的预测分类打分
        overall_p, overall_r, overall_f1 = self.cacl_f1(ents_true_bet, ents_pred_bet) # 在span的始末位置和实体类型方面，都进行确认计算

        # 在最大得分预测实体类型的基础上，添加了阈值作为进一步的判别标准
        threshold_ents_pred_bet = [[ent_inf[:-1] for ent_inf in seq_ents_inf if ent_inf[-1] > self.args.type_threshold] for seq_ents_inf in pred_task_ents_info]
        threshold_p, threshold_r, threshold_f1 = self.cacl_f1(ents_true_bet, threshold_ents_pred_bet)

        ents_true_be = [[(each_ent_true_bet[0], each_ent_true_bet[1]) for each_ent_true_bet in seq_ents_true_bet] for seq_ents_true_bet in ents_true_bet]
        span_p, span_r, span_f1 = self.cacl_f1(ents_true_be, ents_pred_span_be)

        type_p, type_r, type_f1 = self.cacl_f1(tasks_type_trues, tasks_type_preds) # 需要核实，此时的tasks_type_preds是来自于预测的span

        results = {
            "average_loss": average_eval_loss,
            "overall_p": overall_p,
            "overall_r": overall_r,
            "overall_f1": overall_f1,
            "span_p": span_p,
            "span_r": span_r,
            "span_f1": span_f1,
            "type_p": type_p,
            "type_r": type_r,
            "type_f1": type_f1,
            "overall_p_threshold": threshold_p,
            "overall_r_threshold": threshold_r,
            "overall_f1_threshold": threshold_f1,
        }

        logger.info("***** Eval results %s-%s *****", mode, set_type)
        for key in results.keys():
            logger.info("  %s = %.3f", key, results[key] * 100)

        return results, pred_task_ents_info


    def save_model(self, result_dir, fn_prefix, max_seq_len, mode: str = 'overall'):
        """
        该函数的目的在于存储训练好的模型以及模型的相关配置
        :param result_dir:模型的存储地址
        :param fn_prefix: 'begin' ?
        :param max_seq_len:
        :param mode: 训练模式。'all' ?
        :return:
        """
        # 存储模型参数
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model # 需要存储的模型权重
        output_model_file = os.path.join(result_dir, '{}_{}_{}'.format(fn_prefix, mode, WEIGHTS_NAME)) # 模型存储的地址
        torch.save(model_to_save.state_dict(), output_model_file)

        # 存储模型设置
        output_config_file = os.path.join(result_dir, CONFIG_NAME)
        with open(output_config_file, 'w', encoding='utf-8') as f:
            f.write(model_to_save.config.to_json_string())

        # 其他参数设置
        label_map = {i: label for i, label in enumerate(self.label_list, 1)} # 使标签的索引从1开始
        model_config = {
            "bert_model": self.bert_model,
            "do_lower": False,
            "max_seq_length": max_seq_len,
            "num_labels": len(self.label_list) + 1,
            "label_map": label_map,
        }

        # 其他参数保存
        json.dump(
            model_config,
            open(os.path.join(result_dir, '{}-model_config.json'.format(mode)), 'w', encoding='utf-8')
        )

        # 保存类型的向量表示
        if mode == 'type': # 进行第二阶段的类型解码
            joblib.dump(
                self.ent_types, os.path.join(result_dir, 'type_embedding.pkl')
            )

    def get_names(self):
        names = [n for n,p in self.model.named_parameters() if p.requires_grad]
        return names

    def get_params(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return params

    def inner_update(self, data_support, lr_curr, inner_steps, no_grad: bool=False):
        """
        :param data_support: 是一个task里面的数据
        :param lr_curr:
        :param inner_steps: fituning的步数
        :param no_grad:
        :return:
        """
        inner_opt = BertAdam(self.get_optimizer_grouped_parameters(), lr=self.lr_inner)
        self.model.train()

        for i in range(inner_steps):
            inner_opt.param_groups[0]['lr'] = lr_curr
            inner_opt.param_groups[1]['lr'] = lr_curr

            inner_opt.zero_grad()

            # **************************** 此处添加了对比学习所构建的损失 ****************************
            if self.args.use_cl:
                contra_loss = self.model.EntContr_loss(data_support, self.loss_type)
                if contra_loss is not None:
                    contra_loss.backward()
                    inner_opt.step()
                    inner_opt.zero_grad()
            # **************************** 此处添加了对比学习所构建的损失 ****************************

            _, _, loss, type_loss = self.model.forward_ST(
                input_ids=data_support['input_ids'],
                input_mask=data_support['input_mask'],
                segment_ids=data_support['segment_ids'],
                label_ids=data_support['label_ids'],
                # 以上是token级别的操作
                ent_mask=data_support['ent_mask'],
                ent_Ftype_ids=data_support['ent_Ftype_ids'],
                ent_type_mask=data_support['ent_type_mask'],
                entity_types=self.ent_types,
                is_update_type_embedding=True,
                lambda_max_loss=self.args.inner_lambda_max_loss,
                sim_k=self.args.inner_similar_k,
            )# 平常状态下，loss表示监督训练下，token关于entity-span的分类损失

            if loss is None:
                # 对样本进行测试，或者只对entity-span进行type分类训练时
                loss = type_loss
            elif type_loss is not None:
                # ******************************* 两者同时训练时，考虑加权和 ***********************
                # loss = loss + type_loss # 另一个在859行
                loss = 0.7 * loss + 0.3 * type_loss
            if no_grad:
                continue
            loss.backward()
            inner_opt.step()
        # return loss.item()

    def batch_test(self, data):
        """
        :param data: 指的是输入的一个任务的数据
        :return:
        """
        N = data['input_ids'].shape[0] # 该任务中，所有的文本序列的数量
        B = 16 # 这里的batch指的是对一个task中的seq进行batch采样
        BATCH_KEY = [
            "input_ids",
            "input_mask",
            "segment_ids",
            "label_ids",
            "ent_mask",
            "ent_Ftype_ids",
            "ent_type_mask",
        ]
        token_span_logits, ent_type_logits, token_span_loss, ent_type_loss = [], [], 0, 0
        for i in range((N-1) // B + 1): # 多少个batch数量
            tmp_batch_data = {
                ii: jj if ii not in BATCH_KEY else jj[i*B : (i+1)*B]
                for ii, jj in data.items()
            }
            tmp_tsl, tmp_etl, tmp_tsloss, tmp_etloss = self.model.forward_ST(**tmp_batch_data)
            if tmp_tsl is not None:
                token_span_logits.extend(tmp_tsl.detach().cpu().numpy())
            if tmp_etl is not None:
                ent_type_logits.extend(tmp_etl.detach().cpu().numpy())
            if tmp_tsloss is not None:
                token_span_loss += tmp_tsloss
            if tmp_etloss is not None:
                ent_type_loss += tmp_etloss
        return token_span_logits, ent_type_logits, token_span_loss, ent_type_loss

    def eval_typing(self, ent_type_logits, ent_type_mask):
        """
        ********************************************** 该函数用于判断entity-span的具体类型，使用最大相似度来判断，可以考虑KNN ****************************************
        :param ent_type_logits: shape = (seq_num, max_ents, task_type_num), 一个batch中的每个实体属于各个类的可能性
        :param ent_type_mask: shape = (seq_num, max_ent_num, task_type_num)。(max_ent_num, task_type_num)中存在实体的前(ent_num, :)位置为1，其余位置为0
        :return:
        """
        ent_type_mask = ent_type_mask.detach().cpu().numpy()
        if self.is_debug:
            joblib.dump([ent_type_logits, ent_type_mask], "debug/typing.pkl")

        seq_num = len(ent_type_logits)
        batch_size = 16
        res = []
        for i in range((seq_num-1)// batch_size+1):
            ent_type_id = np.argmax(ent_type_logits[i*batch_size: (i+1)*batch_size], -1) # shape = (batch_size, max_ents),元素是最大值的位置
            val_ent_num = ent_type_id.shape[1]
            val_ent_type_mask = ent_type_mask[i * batch_size: (i + 1) * batch_size][:, :val_ent_num, 0] # shape = (batch_size, max_ents),元素为1或0，表示每个句子中实际的实体数量
            res.extend(ent_type_id[val_ent_type_mask == 1])  # 预测的实体类别。该类型是在task中type的顺序id，还需要转换成为具体的类型

        ground = [0] * len(res) # 这是因为所有的entity-span的ground-truth的type-id应该在0的位置
        # return enumerate(res), enumerate(ground) # (data_num, max_ent)表示预测的实体类型；(data_num, )大小的0
        return res, ground

    def decode_span(
        self,
        token_span_logits, # logits # (seq_num, seq_len, BIO_num_labels), 表示每个序列中token作为entity-span的预测概率
        token_span_label, # target # (seq_num, seq_len), 表示每个序列中token作为entity-span的事实标签
        task_types_ids, # types # (seq_num, type_num), 该任务task中所有的实体类型id,
        token_input_mask,# mask # (seq_num, seq_len),对应原始文本序列(填充)的mask
        viterbi_decoder,
    ):
        device = token_span_label.device

        if self.is_debug:
            joblib.dump([token_span_logits, token_span_label, self.label_list], "debug/span.pkl")

        # -------------------------------   通过logits，进行token的标签预测   -------------------------------
        # ******************************* 使用token_span_logits基于维特比进行解码 ************************************
        if viterbi_decoder:
            N = token_span_label.shape[0] # 文本句子序列的数量
            B = 16 # batch_size
            pred_results = []
            for i in range((N-1)//B+1):
                batch_token_span_logits = torch.tensor(token_span_logits[i * B : (i + 1) * B]).to(device)
                if len(batch_token_span_logits.shape) == 2: # 可能在batch的最后剩一条数据
                    batch_token_span_logits = batch_token_span_logits.unsqueeze(0)
                batch_token_span_label = token_span_label[i * B: (i + 1) * B]  # 表示第i个batch的数据

                # *****************************　已经经过了线性分类，为什么此处还需要？*******************************
                batch_token_span_probs = nn.functional.log_softmax(batch_token_span_logits.detach(), dim=-1)  # batch_size x max_seq_len x n_labels
                batch_pred_labels = viterbi_decoder.forward(
                    batch_token_span_probs, token_input_mask[i * B : (i + 1) * B], batch_token_span_label
                )  # batch_size x 每个序列的实际长度(不包含CLS和SEP的标签)

                for ii, jj in zip(batch_pred_labels, batch_token_span_label.detach().cpu().numpy()):  # 对每个句子的预测和实际标签序列进行计算
                    # 考察每一个序列上的预测和真实标签
                    left, right, pad_pre_list = 0, 0, []
                    # 下面代码的目的，主要是为了使得解码后的标签预测序列，能够实现填充。达到和jj一样的填充后长度Max_Seq_Len
                    while right < len(jj) and jj[right] == self.ignore_token_label_id:
                        pad_pre_list.append(-1) # 左侧的填充内容
                        right += 1
                    while left < len(ii):
                        pad_pre_list.append(ii[left])
                        left += 1
                        right += 1
                        while right < len(jj) and jj[right] == self.ignore_token_label_id:
                            pad_pre_list.append(-1)
                            right += 1
                    pred_results.append(pad_pre_list) # result是预测的结果进行串接排布；temp是根据true的前后对pred进行了填充，保持和原来true一样的长度
        else:
            token_span_logits = token_span_logits.detach().cpu().numpy()
            pred_results = np.argmax(token_span_logits, -1)

        token_span_label = token_span_label.detach().cpu().numpy()
        seq_num, seq_len = token_span_label.shape # seq_num, seq_len

        all_pred_ent_bes = []  # List[List[]]，里面那层List[]包括的是每个文本序列中的所有实体的首尾位置。面向预测。

        # 根据具体的entity-span标签，进行实体的获取
        if self.label_list == ["O", "B", "I"]:
            for ii in range(seq_num):
                max_pad = seq_len - 1 # 序列的最后一个token的位置(从0开始)
                while max_pad > 0 and token_span_label[ii][max_pad - 1] == self.pad_token_label_id:
                    max_pad -= 1 # 序列的最后一个不是填充的token位置

                each_seq_ent, idx = [], 0  # each_seq_ent用于存储该文本序列中所有实体的首尾位置
                # 从头开始寻找
                while idx < max_pad:
                    if token_span_label[ii][idx] == self.ignore_token_label_id or pred_results[ii][idx] != 1: # 直到遇见B
                        idx += 1
                        continue
                    ent_b = idx # 由于pred_results[ii][idx] != 1时已经continue，所以此时应该是pred_results[ii][idx] == 1，即B
                    ent_e = ent_b
                    while ent_e < max_pad - 1 and (
                            token_span_label[ii][ent_e + 1] == self.ignore_token_label_id
                            or pred_results[ii][ent_e + 1] in [self.ignore_token_label_id, 2]
                    ):  # 实际上这个判别标准的目的在于直到遇到了O
                        ent_e += 1
                    each_seq_ent.append((ent_b, ent_e))  # 此时的e是一个实体位置的结束部分，即添加了一个entity-span的始末位置。ent_b和ent_e位于entity的正常位置
                    idx = ent_e + 1
                all_pred_ent_bes.append(each_seq_ent)  # list(实体位置的元素), 是预测的结果

        elif self.label_list == ["O", "B", "I", "E", "S"]:
            for ii in range(seq_num):
                max_pad = seq_len - 1 # 序列的最后一个token的位置(从0开始)
                while max_pad > 0 and token_span_label[ii][max_pad - 1] == self.pad_token_label_id:
                    max_pad -= 1 # 序列的最后一个不是填充的token位置

                each_seq_ent, idx = [], 0  # each_seq_ent用于存储该文本序列中所有实体的首尾位置
                # 从头开始寻找
                while idx < max_pad:
                    if token_span_label[ii][idx] == self.ignore_token_label_id or (
                        token_span_label[ii][idx] not in [1, 4]
                    ):
                        idx += 1
                        continue
                    ent_b = idx # 此时ent_b的位置是B或者S
                    ent_iter = ent_b

                    while (
                        ent_iter < max_pad - 1
                        and pred_results[ii][ent_iter] not in [3, 4]
                        and (
                            token_span_label[ii][ent_iter + 1] == self.ignore_token_label_id
                            or pred_results[ii][ent_iter + 1] in [self.ignore_token_label_id, 2, 3]
                        )
                    ): # 满足循环时，ent_iter的位置是I
                        ent_iter += 1

                    if ent_iter < max_pad and pred_results[ii][ent_iter] in [3, 4]:
                        while (
                                ent_iter < max_pad - 1
                                and token_span_label[ii][ent_iter + 1] == self.ignore_token_label_id
                        ):
                            ent_iter += 1
                        each_seq_ent.append((idx, ent_iter))
                    idx = ent_iter + 1
                all_pred_ent_bes.append(each_seq_ent)

        max_type_num = max([len(each_type) for each_type in task_types_ids])
        max_ent_num = max([len(each_sent) for each_sent in all_pred_ent_bes]) # 获取最多的实体数量

        ents_se_mask = np.zeros((seq_num, max_ent_num, seq_len), np.int8) # 实体在序列中相应的[begin: end]位置都为1
        ents_type_mask = np.zeros((seq_num, max_ent_num, max_type_num), np.int8) # batch_size, max_ent, type_num
        ents_type_ids = np.zeros((seq_num, max_ent_num, max_type_num), np.int) # batch_size, max_ent, type_num

        for ii in range(seq_num):
            for idx, (s, e) in enumerate(all_pred_ent_bes[ii]):
                ents_se_mask[ii][idx][s : e + 1] = 1 # 对每个实体，在序列中构造出他的mask
            types_set = task_types_ids[ii] # 第ii条数据的实体类型集合
            if len(all_pred_ent_bes[ii]): # 实体的数量，或者叫做最大实体数
                ents_type_ids[ii, : len(all_pred_ent_bes[ii]), : len(types_set)] = [types_set] * len(all_pred_ent_bes[ii])
            ents_type_mask[ii, : len(all_pred_ent_bes[ii]), : len(types_set)] = np.ones((len(all_pred_ent_bes[ii]), len(types_set)))

        return (
            torch.tensor(ents_se_mask).to(device), # (seq_num, max_ent, seq_len), 对每个实体，在独立的序列中构造出他的mask
            torch.tensor(ents_type_ids, dtype=torch.long).to(device), # (seq_num, max_ent, task_type_num), 每个ent对应该task中的所有entity type
            torch.tensor(ents_type_mask).to(device), # (seq_num, max_ent, task_type_num)。每个句子所对应的矩阵(max_ent, task_type_num)中，[:ent_num, :type_num]为1
            all_pred_ent_bes, # list(list(实体位置的元组)), 是预测的结果
            task_types_ids, # list(list(该task中所有的type的id))
        )

    def load_weights(self, names, params):
        model_params = self.model.state_dict()
        for name, param in zip(names, params):
            model_params[name].data.copy_(param.data)

    def decode_entity(self, ents_type_logits, ents_pred_be, task_type_ids, true_entities_list):
        # ents_type_logits，是在预测的span的基础上进行计算的，指的是ents_pred_be对应的span
        # true_entities_list: 序列中实体的列表[(ent_s, ent_e, ent_t), ...], ent_s和ent_e是在添加了['CLS']的前提下的始末位置，ent_t是实体类型的id
        if self.is_debug:
            joblib.dump([ents_type_logits, ents_pred_be, task_type_ids, true_entities_list], "debug/e.pkl")

        true_ents_info, pred_ents_info = true_entities_list, []
        seq_num = len(ents_type_logits)

        for seq_id in range(seq_num):
            each_seq_ents_info = []
            each_seq_ents_pred_be = ents_pred_be[seq_id]
            each_seq_task_type_ids = task_type_ids[seq_id]

            for ent_id in range(len(each_seq_ents_pred_be)):
                each_ent_logit = ents_type_logits[seq_id][ent_id, : len(each_seq_task_type_ids)]
                each_seq_ents_info.append( # 每个元素包括了实体的[(实体预测的始末位置，预测的实体类型，实体的预测分类打分),...]
                    (*each_seq_ents_pred_be[ent_id], each_seq_task_type_ids[np.argmax(each_ent_logit)], each_ent_logit[np.argmax(each_ent_logit)]))

            pred_ents_info.append(each_seq_ents_info)

        return true_ents_info, pred_ents_info

    def cacl_f1(self, true_ents, pred_ents):
        tp, fp, fn = 0, 0, 0
        for ii, jj in zip(true_ents, pred_ents):
            ii, jj = set(ii), set(jj)
            same = ii - (ii - jj)
            tp += len(same)
            fn += len(ii - jj)
            fp += len(jj - ii)
        p = tp / (fp + tp + 1e-10)
        r = tp / (fn + tp + 1e-10)
        return p, r, 2 * p * r / (p + r + 1e-10)


# ********************************************* 以上主要是关于初始化和测试，下面面向训练 *****************************************

    def load_gradients(self, names, grads):
        model_params = self.model.state_dict(keep_vars=True)
        for n, g in zip(names, grads):
            if model_params[n].grad is None:
                continue
            model_params[n].grad.data.add_(g.data) # 进行加速

    def get_learning_rate(self, lr, progress, warmup, schedule='linear'):
        if schedule == 'linear':
            if progress < warmup:
                lr *= progress / warmup
            else:
                lr *= max((progress-1.0)/(warmup-1.0), 0.0)
        return lr

    def forward_supervise(self, batch_query, batch_support, progress, inner_steps):
        # 该函数依次在train数据集的support和query上进行finetune
        span_losses, type_losses = [], []
        task_num = len(batch_query)

        # ***************************** 在训练样本的eval部分进行测试和finetuning *****************************
        for task_id in range(task_num):
            _, _, loss, type_loss = self.model.forward_ST(
                input_ids=batch_query[task_id]['input_ids'],
                input_mask=batch_query[task_id]['input_mask'],
                segment_ids=batch_query[task_id]['segment_ids'],
                label_ids=batch_query[task_id]['label_ids'],
                # 以上是token级别的操作
                ent_mask=batch_query[task_id]['ent_mask'],
                ent_Ftype_ids=batch_query[task_id]['ent_Ftype_ids'],
                ent_type_mask=batch_query[task_id]['ent_type_mask'],
                entity_types=self.ent_types,
                lambda_max_loss=self.args.lambda_max_loss,
            )# 平常状态下，loss表示监督训练下，token关于entity-span的分类损失

            if loss is not None:
                span_losses.append(loss.item())
            if type_loss is not None:
                type_losses.append(type_loss.item())

            if loss is None:
                loss = type_loss
            elif type_loss is not None:
                loss = loss + type_loss

            loss.backward()
            self.opt.step()
            self.scheduler.step()
            self.model.zero_grad()

        # ***************************** 在训练样本的support部分进行测试和finetuning *****************************
        for task_id in range(task_num):
            _, _, loss, type_loss = self.model.forward_ST(
                input_ids=batch_support[task_id]['input_ids'],
                input_mask=batch_support[task_id]['input_mask'],
                segment_ids=batch_support[task_id]['segment_ids'],
                label_ids=batch_support[task_id]['label_ids'],
                # 以上是token级别的操作
                ent_mask=batch_support[task_id]['ent_mask'],
                ent_Ftype_ids=batch_support[task_id]['ent_Ftype_ids'],
                ent_type_mask=batch_support[task_id]['ent_type_mask'],
                entity_types=self.ent_types,
                lambda_max_loss=self.args.inner_lambda_max_loss,
            )# 平常状态下，loss表示监督训练下，token关于entity-span的分类损失

            if loss is not None:
                span_losses.append(loss.item())
            if type_loss is not None:
                type_losses.append(type_loss.item())
            if loss is None:
                loss = type_loss
            elif type_loss is not None:
                loss = loss + type_loss

            loss.backward()
            self.opt.step()
            self.scheduler.step()
            self.model.zero_grad()

        return (
            np.mean(span_losses) if span_losses else 0,
            np.mean(type_losses) if type_losses else 0,
        ) # 对support中和eval中的所有span_loss以及type_loss求平均

    def forward_meta(self, batch_query, batch_support, progress, inner_steps):
        names = self.get_names()
        params = self.get_params()
        weights = deepcopy(params)

        meta_grad = []
        span_losses, type_losses = [], []

        task_num = len(batch_query)
        lr_inner = self.get_learning_rate(
            self.lr_inner, progress, self.warmup_prop_inner
        )

        # 计算每一个任务的元梯度
        for task_id in range(task_num): # 每个task认为是一个episode
            # inner_update
            self.inner_update(batch_support[task_id], lr_inner, inner_steps=inner_steps)

            # meta-update

            # **************************** 此处添加了对比学习所构建的损失 ****************************
            if self.args.use_cl:
                contra_loss = self.model.EntContr_loss(batch_query[task_id], self.loss_type)
                if contra_loss:
                    contra_loss.backward()
                    self.opt.step()
                    # self.scheduler.step()
                    self.model.zero_grad()
            # **************************** 此处添加了对比学习所构建的损失 ****************************

            _, _, loss, type_loss = self.model.forward_ST(
                input_ids=batch_query[task_id]['input_ids'],
                input_mask=batch_query[task_id]['input_mask'],
                segment_ids=batch_query[task_id]['segment_ids'],
                label_ids=batch_query[task_id]['label_ids'],
                # 以上是token级别的操作
                ent_mask=batch_query[task_id]['ent_mask'],
                ent_Ftype_ids=batch_query[task_id]['ent_Ftype_ids'],
                ent_type_mask=batch_query[task_id]['ent_type_mask'],
                entity_types=self.ent_types,
                lambda_max_loss=self.args.inner_lambda_max_loss,
            )  # 平常状态下，loss表示监督训练下，token关于entity-span的分类损失

            if loss is not None:
                span_losses.append(loss.item())
            if type_loss is not None:
                type_losses.append(type_loss.item())
            if loss is None:
                loss = type_loss
            elif type_loss is not None:
                # loss = loss + type_loss
                loss = 0.7* loss + 0.3 * type_loss # 另一处在490行
            grad = torch.autograd.grad(loss, params, allow_unused=True) # meta-update的核心所在
            meta_grad.append(grad)

            self.load_weights(names, weights)

        # 累计所有任务的梯度来给参数梯度
        self.opt.zero_grad()

        # 类似于backward()操作
        for g in meta_grad:
            self.load_gradients(names, g)
        self.opt.step()
        self.scheduler.step()

        return (
            np.mean(span_losses) if span_losses else 0,
            np.mean(type_losses) if type_losses else 0,
        )

    # 以下为评估阶段
    def write_result(self, words, y_true, y_pred, tmp_fn):
        assert len(y_pred) == len(y_true)
        with open(tmp_fn, 'w', encoding='utf-8') as fw:
            for i, sent in enumerate(y_true):
                for j, word in enumerate(sent):
                    fw.write('{} {} {}\n'.format(words[i][j], word, y_pred[i][j])) # 每个token，以及其实际标签和预测标签
                fw.write('\n')

