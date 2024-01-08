#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/11/2 上午9:33
# @Author  : PeiP Liu
# @FileName: main.py
# @Software: PyCharm

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
import joblib

from Model.learner import Learner
from Data.utils import *
from Data.EntityTypes import EntityTypes
from Data.Corpus import Corpus
from Functions.evaluate import *
from Functions.train import train

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    def my_bool(s):
        return s != "False"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FewNERD') # 数据集的设置。也要做跨域的数据集(命名为CrossDomain/CrossDomain2):Cross-Dataset [Hou et al., 2020](https://www.aclweb.org/anthology/2020.acl-main.128) 。
    parser.add_argument('--N', type=int, default=5) # N-way K-shot的设置
    parser.add_argument('--K', type=int, default=1) # N-way K-shot的设置
    parser.add_argument('--mode', type=str, default="intra") # 选择intra或者inner数据集
    parser.add_argument('--test_only', action="store_true") #使用python test.py --test_only来触发返回True。表示只进行测试
    parser.add_argument('--convert_bpe', action="store_true") # 同理触发convert_bpe。第一步不能有，因为需要产生预测数据才能计算
    parser.add_argument('--tagging_scheme', type=str, default='BIO', help='BIO or BIEOS') # 采用BIO/BIEOS的方式来标注实体跨度
    parser.add_argument('--data_path', type=str, default="Dataset/FewNERD") # 数据的来源地址, Dataset/Domain、Dataset/Domain2.注意utils文件中的get_data_path()函数
    parser.add_argument('--result_dir', type=str, default='', help='the path to save results') # 测试结果的存储地址
    parser.add_argument('--model_dir', type=str, default='', help='the path for saving trained model') # 存储训练好的模型的地址

    # the settings for meta-test
    parser.add_argument('--lr_finetune', type=float, default=3e-5, help='finetune learning rate, used in [test_meta] and [k-shot]') # 测试评估专用：微调时的学习率.用于在test中的support进行微调模型时
    parser.add_argument('--max_ft_steps', type=int, default=3) #　测试评估专用：inner_update更新的次数
    parser.add_argument('--max_type_ft_steps', type=int, default=2) # inner_update更新的次数

    # the settings for meta-train
    parser.add_argument('--inner_steps', type=int, default=2) # 每inner_steps个inner_update，能够实现meta-train中的一次meta-update
    parser.add_argument('--inner_size', type=int, default=32) # 表示一次meta-update中的数据量，也就是task的任务量.如果内存不够，适当减小
    parser.add_argument('--lr_inner', type=float, default=3e-5) # inner学习中的学习率
    parser.add_argument('--lr_meta', type=float, default=3e-5) # ？？？？

    parser.add_argument('--max_meta_steps', type=int, default=5001) # ？？？
    parser.add_argument('--eval_every_meta_steps', type=int, default=500)
    parser.add_argument('--warmup_prop_inner', type=float, default=0.1)
    parser.add_argument('--warmup_prop_meta', type=float, default=0.1)

    # bert模型相关参数
    parser.add_argument('--freeze_layer', type=int, default=0)
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--bert_model', type=str, default='Bert/bert-base-uncased') # bert模型的位置
    parser.add_argument('--cache_dir', type=str, default='Bert/pretrained_bert/')

    # 实验系统设置相关的信息
    parser.add_argument('--viterbi', type=str, default='hard') # 采用hard/soft/None，理论上，训练时为soft，测试和验证时为hard
    parser.add_argument('--concat_types', type=str, default='past') # 采用past/before/None的方式连接task types到文本序列中
    parser.add_argument('--seed', type=int, default=667) # 复现结果的随机种子
    parser.add_argument('--gpu_device', type=int, default=0) # 默认使用0号GPU
    parser.add_argument('--py_alias', type=str, default='python')

    parser.add_argument('--types_path', type=str, default="Dataset/TypeData/entity_types.json") # 所有实体类型的存储位置
    parser.add_argument('--negative_types_number', type=int, default=4) # 负的实体类型的数量
    parser.add_argument('--negative_mode', type=str, default='batch') # 负实体类型的表现形式
    parser.add_argument('--types_mode', type=str, default='cls') # 使用什么形式表示类的初始向量
    parser.add_argument('--name', type=str, default='') # 进行实验的名称
    parser.add_argument('--debug', action="store_true")

    parser.add_argument('--init_type_embedding_from_bert', action="store_true",) # 使用BERT来初始化表示类型的向量
    parser.add_argument('--use_classify', action="store_true") # 使用原始BERT中的线性分类器
    parser.add_argument('--distance_mode', type=str, default='cos') # 计算向量间相似性的方式
    parser.add_argument('--similar_k', type=float, default=10) # 计算几个最近邻的相似性

    parser.add_argument('--shared_bert', type=my_bool, default=True)
    parser.add_argument('--train_mode', type=str, default='add') # 存在add、span、type等形式，训练模式为add的时候，是联合训练
    parser.add_argument('--eval_mode', type=str, default='add') # 存在add、two-stage等形式
    parser.add_argument('--type_threshold', type=float, default=2.5) # 设定解码得分的阈值


    parser.add_argument('--lambda_max_loss', type=float, default=2.0)
    parser.add_argument('--inner_lambda_max_loss', type=float, default=2.0)
    parser.add_argument('--inner_similar_k', type=float, default=10) # inner更新时计算几个最近邻的相似性
    parser.add_argument('--ignore_eval_test', action="store_true") # 是否要在测试中进行评估
    parser.add_argument('--nouse_inner_ft', action="store_true") # 此步骤，理论上可有可无。但应该测试他的True和False，实际上是进行了eval-support的finetune
    parser.add_argument('--use_supervise', action="store_true") # 两种不同的模型训练方式。如果不选择use_supervise，那么就是原始论文中的meta训练方式。

    parser.add_argument('--link_temperature', type=float, default=0.41)  # KNN参数, 0.01 0.11 0.21 0.31 0.41 0.51 0.61 0.71 0.81 0.91
    parser.add_argument('--link_ratio', type=float, default=0.31)  # KNN参数, 0.01 0.11 0.21 0.31 0.41 0.51 0.61 0.71 0.81 0.91
    parser.add_argument('--topk', type=float, default=10)  # KNN参数
    parser.add_argument('--use_cbknn', action="store_true")  # 在训练时，选择最优模型也使用了KNN。use_cbknn && use_knn才会使用
    parser.add_argument('--use_knn', action="store_true")  # 使用KNN进行测试融合。
    parser.add_argument('--use_cl', action="store_true")  # 是否选择对比学习进行实体类别分离．KNN和cl都是在type训练和only test时使用

    args = parser.parse_args()
    args.negative_types_number = args.N - 1 # 负样本类别数量

    if 'Domain' in args.dataset:
        args.types_path = 'Dataset/TypeData/entity_types_domain.json' # 把原始的初始化类型改成domain相关的

    # setup random seed
    set_seed(args.seed, args.gpu_device)

    # set up GPU device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.gpu_device)

    # **************** 针对具体数据集具体模式和选项的结果文件夹构建 ****************
    top_dir = 'Results/{}/{}/model-{}-{}'.format(args.dataset, args.mode, args.N, args.K)  # 模型的模式
    # args.name = "{}-k_{}_{}_{}_{}_max_loss-{}_{}_{}".format(
    #     args.similar_k, args.eval_every_meta_steps, args.inner_steps, args.inner_size, args.max_ft_steps, args.lambda_max_loss, args.inner_lambda_max_loss, args.tagging_scheme)

    tempMR_dir = '{}/{}-innerSteps_{}-innerSize_{}-lrInner_{}-lrMeta_{}-maxSteps_{}-seed_{}/{}'.format(
        top_dir,
        args.bert_model,
        args.inner_steps,
        args.inner_size,
        args.lr_inner,
        args.lr_meta,
        args.max_meta_steps,
        args.seed,
        "name_{}".format(args.name) if args.name else "",
    )

    if not os.path.exists(tempMR_dir):
        os.makedirs(tempMR_dir)

    # ******************* 设置日志对象，并进行部分信息打印 *******************
    # setup logger settings
    if args.test_only: # 只测试，说明训练好了模型后就行测试就好
        args.model_dir = tempMR_dir
        fh = logging.FileHandler(
            '{}/log-test-ftLr_{}-ftSteps_{}.txt'.format(
                args.model_dir, args.lr_finetune, args.max_ft_steps,
            ) # 建立一个日志文件
        )
    else:
        args.result_dir = tempMR_dir
        fh = logging.FileHandler('{}/log-training.txt'.format(args.result_dir))
        # dump args
        with Path('{}/args-train.json'.format(args.result_dir)).open('w', encoding='utf-8') as fw:
            json.dump(vars(args), fw, indent=4, sort_keys=True) # 将配置参数写入文件中

    if args.debug:
        os.makedirs('debug', exist_ok=True)

    logger = logging.getLogger() # 构建日志处理对象
    logger.setLevel(logging.INFO) # 可以不设，默认是WARNING级别

    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    ) # 构建日志内容对象
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter) # 设置文件的log格式

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter) # 构建流处理对象

    logger.addHandler(ch)
    logger.addHandler(fh) # 将所有的对象添加到日志处理对象中

    args.device = device
    logger.info('Using Device {}'.format(device)) # 输出日志信息

    # ******************* 进行类型对象的构建，并初始化类型向量 *******************
    args.ent_types = EntityTypes(
        args.types_path, args.negative_mode # types_path, negative_mode
    )
    args.ent_types.building_types_embedding( # 初始化构建类型的向量
        args.bert_model, # 选择的bert模型的类型
        True,
        args.device, # 模型加载使用的gpu信息
        args.types_mode, # 通过bert获取type embedding的方式
        args.init_type_embedding_from_bert, #
    )

    label_list = get_label_list(args) # 字符标签构造的列表
    if args.convert_bpe: # 用于预测后，将标签转换到词汇上面。可以不使用。
        convert_bpe(args, logger, label_list)
    elif args.test_only:
        if args.model_dir == '':
            # 测试时，应该已经产生了训练好的模型。因此，模型地址不应该为空。
            raise ValueError('Null Model Directionary...')
        evaluate(args, logger, label_list)
    else:
        if args.model_dir != '': #　执行训练任务时，该文件夹应该为空
            raise ValueError("Model directory should be NULL!")
        train(args, logger, label_list)

