#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/11/17 下午8:09
# @Author  : PeiP Liu
# @FileName: utils.py
# @Software: PyCharm
import random
import json
import os
import joblib

import numpy as np
import torch
from Data.Corpus import Corpus

def set_seed(seed, gpu_device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if gpu_device > -1:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_file(file_path: str, mode: str = 'list-strip'):
    if not os.path.exists(file_path):
        return [] if not mode else ''

    with open(file_path, 'r', encoding='utf-8', newline='\n') as f:
        if mode == 'list-strip':
            data = [ii.strip() for ii in f.readlines()]
        elif mode == 'str':
            data = f.read()
        elif mode == 'list':
            data = list(f.readlines())
        elif mode == 'json':
            data = json.loads(f.read())
        elif mode =='json-list':
            data = [json.loads(ii) for ii in f.readlines()] # loads是文件流转成字典
    return data

def get_label_list(args): # 字符标签构造的列表
    # present the dataset
    if args.tagging_scheme == "BIEOS":
        label_list = ["O", "B", "I", "E", "S"]
    elif args.tagging_scheme == "BIO":
        label_list = ["O", "B", "I"]
    else:
        label_list = ["O", "I"]
    return label_list

def get_data_path(args, opt_mode):
    # 返回数据的地址，包括N, K和train/dev/test
    assert args.dataset in ['FewNERD', 'CrossDomain', 'CrossDomain2'], 'Dataset {} Not support'.format(args.dataset)
    if args.dataset == 'FewNERD':
        return os.path.join(args.data_path, args.mode, '{}_{}_{}.jsonl'.format(opt_mode, args.N, args.K))
    elif args.dataset == 'CrossDomain':
        if opt_mode == 'dev':
            opt_mode = 'valid'
        text = '_shot_5' if args.K == 5 else ''
        replace_text = '-' if args.K == 5 else '_'
        return os.path.join('ACL2020data', "xval_ner{}".format(text),
                "ner_{}_{}{}.json".format(opt_mode, args.N, text).replace("_", replace_text)
                ) # 具体数据下载后再看
    elif args.dataset == 'CrossDomain2':
        if opt_mode == 'train':
            return os.path.join('domain2', '{}_10_5.json'.format(opt_mode))
        return os.path.join('domain2', '{}_{}_{}.json'.format(opt_mode, args.mode, args.K))

def convert_bpe(args, logger, label_list):
    logger.info('************** Scheme: Convert BPE **************')
    os.makedirs('preds', exist_ok=True)

    def convert_base(opt_mode):
        data_path = get_data_path(args, opt_mode) # 获取的数据地址
        corpus = Corpus( # 构造一个数据集类型对象
            logger,
            data_path, # 数据集的来源地址
            args.bert_model,
            args.max_seq_len,
            label_list, # 字符标签构造的列表OBI
            args.entity_types, # EntityTypes类型的对象
            do_lower_case = True,
            shuffle = False,
            tagging = args.tagging_scheme, # 'OBI or OBIES'，默认OBI
            viterbi = args.viterbi, # viterbi的类型，默认hard
            concat_types = args.concat_types, # 如何将类向量进行表示，衔接进入
            dataset = args.dataset, # str类型，数据集名称
            device = args.device
        )
        for seed in [171, 354, 550, 667, 985]:
            path = os.path.join(args.model_dir, "all_{}_preds.pkl".format(opt_mode if opt_mode == 'test' else 'valid')).replace("171", str(seed))
            data = joblib.load(path)

            # 整理预测数据标签和真实标签
            if len(data) == 3:
                pred_spans = data[-1]
            else:
                pred_spans = [[jj[:-2] for jj in ii] for ii in data[-1]]
            target_spans = [[jj[:-1] for jj in ii] for ii in data[0]]

            pred_res = corpus.decode_bpe_index(pred_spans) # pred_spans是预测的实体位置结果(以token为单元)，遵循token_nums的形式。
            true_res = corpus.decode_bpe_index(target_spans)# target_spans是实际的实体位置结果(以token为单元)，遵循token_nums的形式。
            with open("preds/{}-{}way{}shot-seed{}-{}_pred.jsonl".format(args.mode, args.N, args.K, seed, opt_mode), 'w') as f:
                json.dump(pred_res, f)
            with open("preds/{}-{}way{}shot-seed{}-{}_true.jsonl".format(args.mode, args.N, args.K, seed, opt_mode), 'w') as f:
                json.dump(true_res, f) # 解码后的true_res是span中实体词的位置，包括['CLS']为开始。

    convert_base('dev')
    convert_base('test')