#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/11/24 下午1:44
# @Author  : PeiP Liu
# @FileName: evaluate.py
# @Software: PyCharm
from Data.utils import *
from Model.learner import Learner
from Model.viterbi import ViterbiDetector

def evaluate(args, logger, label_list):
    logger.info('********** Scheme: Meta Test **********')
    valid_data_path = get_data_path(args, 'dev')
    valid_corpus = Corpus(
        logger,
        valid_data_path,
        args.bert_model,
        args.max_seq_len,
        label_list,
        args.ent_types,
        do_lower_case=True,
        shuffle=False,
        tagging=args.tagging_scheme,
        viterbi=args.viterbi, # 测试集的viterbi是hard，说明不需要进行调整
        concat_types=args.concat_types,
        dataset=args.dataset,
        device=args.device
    )

    test_data_path = get_data_path(args, 'test')
    test_corpus = Corpus(
        logger,
        test_data_path,
        args.bert_model,
        args.max_seq_len,
        label_list,
        args.ent_types,
        do_lower_case=True,
        shuffle=False,
        tagging=args.tagging_scheme,
        viterbi=args.viterbi, # 测试集的viterbi是hard，说明不需要进行调整
        concat_types=args.concat_types,
        dataset=args.dataset,
        device=args.device
    )

    learner = Learner(
        args.bert_model,
        label_list,
        args.freeze_layer,
        logger,
        args.lr_meta,
        args.lr_inner,
        args.warmup_prop_meta,
        args.warmup_prop_inner,
        args.max_meta_steps,
        model_dir = args.model_dir,
        py_alias = args.py_alias,
        args = args,
    )

    logger.info("********** Scheme: evaluate - [valid] **********")
    _, _ = test(args, learner, logger, valid_corpus, "valid")

    logger.info("********** Scheme: evaluate - [test] **********")
    _, _ = test(args, learner, logger, test_corpus, "test")

def test(args, learner, logger, corpus, types: str):
    """
    :param args: 参数配置
    :param learner: 模型学习器、优化器的类对象
    :param logger: 日志对象
    :param corpus: 验证集或者测试集的数据集对象
    :param types: "valid"或者"test"
    :return:
    """
    if corpus.viterbi != 'none': # 实际上，参与使用的话，本身是soft或者hard
        id2label = corpus.id2label # 训练集上的span标签
        transition_matrix = corpus.transition_matrix
        if args.viterbi == 'soft':
            label_list = get_label_list(args)
            train_data_path = get_data_path(args, 'train') # 从训练数据处获取viterbi的转换矩阵
            train_corpus = Corpus(
                logger,
                train_data_path,
                args.bert_model,
                args.max_seq_len,
                label_list,
                args.ent_types, # EntityTypes类型对象
                do_lower_case=True,
                shuffle=True,
                tagging=args.tagging_scheme, # 实体span的标注方式，BIO/BIOES
                viterbi='soft', # 训练集的viterbi是soft，说明需要进行调整
                device=args.device,
                concat_types=args.concat_types,
                dataset=args.dataset # 数据集的名称
            ) # 构造训练数据集的对象
            id2label = train_corpus.id2label
            transition_matrix = train_corpus.transition_matrix
        viterbi_decoder = ViterbiDetector(id2label, transition_matrix)
    else:
        viterbi_decoder = None

    true_result, pred_result = learner.evaluate_meta_(
        corpus,
        logger,
        lr = args.lr_finetune, # 测试评估专用：主要用于meta-testing中对support的finetune
        steps = args.max_ft_steps, # 测试评估专用：
        mode = args.mode, # 选择intra或者inner
        set_type = types, # "valid"或者"test"
        type_steps=args.max_type_ft_steps,
        viterbi_decoder=viterbi_decoder # 默认是不会为None的，因此存在实际的viterbi
    )
    return true_result, pred_result
