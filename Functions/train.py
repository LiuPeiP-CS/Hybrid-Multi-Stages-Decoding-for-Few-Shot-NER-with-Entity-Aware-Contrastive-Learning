#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/12/7 下午11:23
# @Author  : PeiP Liu
# @FileName: train.py
# @Software: PyCharm

import torch
import time
from Functions.evaluate import *

def train(args, logger, label_list):
    logger.info("********** Scheme: Meta Learning **********")
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

    if args.debug: # 测试调整模型，核心意义不大。就是训练并调整自己.默认应该为False
        test_corpus = valid_corpus
        train_corpus = valid_corpus
    else: # 正常训练模型
        train_data_path = get_data_path(args, "train")
        train_corpus = Corpus(
            logger,
            train_data_path,
            args.bert_model,
            args.max_seq_len,
            label_list,
            args.ent_types,
            do_lower_case=True,
            shuffle=True,
            tagging=args.tagging_scheme,
            # viterbi=args.viterbi,
            device=args.device,
            concat_types=args.concat_types,
            dataset=args.dataset,
        )
        if not args.ignore_eval_test: # 每轮训练后，都要进行测试。默认为执行
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
    ) # 构建学习器

    if "embedding" in args.concat_types: # 在训练时，基本不会出现。因为concat_types的默认值为before或者past
        replace_type_embedding(learner, args, logger)

    tim = time.time()
    f1_valid_best = {item: -1.0 for item in ["overall", "type", "span"]}

    best_step, protect_step = -1.0, 100 if args.train_mode != "type" else 50

    # 训练的epoch数量.理论上，每个epoch都应该训练所有的数据，然而这里却训练了batch的数据？？？
    for step in range(args.max_meta_steps):
        progress = 1.0 * step / args.max_meta_steps

        # 指的是训练样本数据中一个batch的task的结果。每次随机选出来32个
        batch_train_query, batch_train_support = train_corpus.get_batch_meta(
            batch_size=args.inner_size
        )  # (batch_size=32),

        # 两种不同的训练方式。方式的主要不同在于，supervise是分别完全训练support的batch和query的batch；而meta是按照一个task(即一个episode)进行同时连续训练
        if args.use_supervise:
            span_loss, type_loss = learner.forward_supervise( # 针对训练集的query和support进行训练
                batch_train_query,
                batch_train_support,
                progress=progress,
                inner_steps=args.inner_steps,
            )
        else: # 论文描述是此种训练方式
            span_loss, type_loss = learner.forward_meta( # 只针对训练集的query进行训练
                batch_train_query,
                batch_train_support,
                progress=progress,
                inner_steps=args.inner_steps,
            )

        # # 查看GPU的使用情况
        # current_gpu_index = torch.cuda.current_device()
        # used_memory = torch.cuda.memory_allocated(current_gpu_index) / (1024 ** 2)
        # logger.info("一轮Meta训练之后，已使用的GPU显存：{:.2f} MB".format(used_memory))

        # ******************************************** 训练过程中的细节信息展示 ***********************************
        if step % 40 == 0: # 每训练40个epoch进行细节输出
            logger.info(
                "Step: {}/{}, span loss = {:.6f}, type loss = {:.6f}, time = {:.2f}s.".format(
                    step, args.max_meta_steps, span_loss, type_loss, time.time() - tim
                )
            )

        if step % args.eval_every_meta_steps == 0 and step > protect_step:
            # 每训练eval_every_meta_steps个epoch，进行一次eval，并存储相应的最优模型
            logger.info("********** Scheme: evaluate - [valid] **********")
            result_valid, _ = test(args, learner, logger, valid_corpus, "valid")

            overall_f1_valid = result_valid["overall_f1"]
            is_best = False
            if overall_f1_valid > f1_valid_best["overall"]:
                logger.info("===> Best Valid F1: {}".format(overall_f1_valid))
                logger.info("  Saving model...")
                learner.save_model(args.result_dir, "en", args.max_seq_len, "overall")
                f1_valid_best["overall"] = overall_f1_valid
                best_step = step
                is_best = True

            if result_valid["span_f1"] > f1_valid_best["span"] and args.train_mode != "type":
                f1_valid_best["span"] = result_valid["span_f1"]
                learner.save_model(args.result_dir, "en", args.max_seq_len, "span")
                logger.info("Best Span Store {}".format(step))
                is_best = True
            if result_valid["type_f1"] > f1_valid_best["type"] and args.train_mode != "span":
                f1_valid_best["type"] = result_valid["type_f1"]
                learner.save_model(args.result_dir, "en", args.max_seq_len, "type")
                logger.info("Best Type Store {}".format(step))
                is_best = True

            if is_best and not args.ignore_eval_test:
                logger.info("********** Scheme: evaluate - [test] **********")
                result_test, _ = test(args, learner, logger, test_corpus, "test")

                F1_test = result_test["overall_f1"]
                logger.info("Best Valid F1: {}, Step: {}".format(f1_valid_best, best_step))
                logger.info("Test F1: {}".format(F1_test))


def replace_type_embedding(learner, args, logger):
    logger.info("********** Replace trained type embedding **********")
    # **************************************************** 此处何来type_embedding.pkl文件？ **********************************************************
    entity_types = joblib.load(os.path.join(args.result_dir, "type_embedding.pkl"))
    types_num, _ = entity_types.types_embedding.weight.data.shape  # 分别表示type的数量和特征维度
    for type_id in enumerate(types_num):
        learner.model.embeddings.word_embeddings.weight.data[type_id+1] = entity_types.types_embedding.weight.data[type_id]