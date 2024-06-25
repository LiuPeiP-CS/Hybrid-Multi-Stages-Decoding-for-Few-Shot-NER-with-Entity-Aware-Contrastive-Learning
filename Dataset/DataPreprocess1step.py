#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2024/1/12 上午9:33
# @Author  : PeiP Liu
# @FileName: main.py
# @Software: PyCharm

import json
import os
import random


def load_file(orig_data_path, ent_types_path):
    orig_file = open(orig_data_path, 'r', encoding='utf-8')

    sentences4labels = []
    sentence = []
    sentence_label = []
    eiter = 0

    entity_types = set()
    for line in orig_file:
        eiter += 1
        line = line.strip()
        if line == '': # 如果遇到了空行，说明一个句子已经结束了
            if not sentence:
                continue
            sentences4labels.append((sentence, sentence_label))
            if len(sentence) != len(sentence_label):
                print("The length of sentence is not equal to the sentence-label in {}!".format(eiter))
            sentence = []
            sentence_label = []
        else: # 正常处理的行信息
            word_label = line.split()
            if len(word_label) == 2:
                label = word_label[-1].strip() # 是token的标签
            elif len(word_label) == 1: # 经过检查，是只有标注为O的空信息
                label = 'O'
            else:
                print("{} line: {}".format(eiter, line))

            if '-' in label:
                label = label[2:] # 实体的B-/I-/E-/S- LABEL的标签
            elif label != 'O': # 既不包含-，也不是O，一定有问题
                print("{}: {}".format(eiter, line))

            sentence.append(word_label[0])  # 实体token
            entity_types.add(label) # 构建所有实体类型集合
            sentence_label.append(label) # 一条数据的标签序列

    # 以下内容为存储APTNER的实体类型
    ent_types_dict = {'entity_types': list(entity_types)}
    with open(ent_types_path, 'w', encoding='utf-8') as outfile:
        json.dump(ent_types_dict, outfile, indent=4)

    return sentences4labels, entity_types


def type_count(all_data, all_types):
    type_cnum = dict()
    for t in all_types:
        if t != 'O':
            type_cnum[t] = 0 # 初始化实体类型及相应的数据数量

    def delrep(each_datalabel):
        # 对连续重复的字符串去重
        new_label_seq = []
        current_label = ''
        label_iter = 0
        while label_iter < len(each_datalabel):
            if each_datalabel[label_iter] != current_label:
                new_label_seq.append(each_datalabel[label_iter])
                current_label = each_datalabel[label_iter]
            label_iter += 1
        return new_label_seq

    for (_, each_datalabel) in all_data:
        nlabel_seq = delrep(each_datalabel)
        for eachent in nlabel_seq:
            if eachent in type_cnum:
                type_cnum[eachent] += 1

    print(type_cnum)


def build_type4data(sentences4labels, entity_types):
    entity_type4sentslabels = dict() # 构建每个实体类别下面的所有序列
    for each_entity_type in entity_types:
        entity_type4sentslabels[each_entity_type] = []
        for each_sentslabels in sentences4labels:
            assert len(each_sentslabels[0]) == len(each_sentslabels[1])
            if each_entity_type in each_sentslabels[1]: # 如果该实体在某序列标签中，则认为该实体类型
                entity_type4sentslabels[each_entity_type].append(each_sentslabels)
    return entity_type4sentslabels


def build_dataset(entity_type4sentslabels, entity_types, types_num, data_num, N, K, data_mode):
    """
    :param entity_type4sentslabels: 每个实体类对应的数据
    :param entity_types: 总的实体类型
    :param types_num: 实体类型的数量
    :param data_num: 需要构建的数据量
    :param N: N-way
    :param K: K-shot
    :return:
    """
    dataset = []
    for data_iter in range(data_num):
        each_task = {"support": {"word": [], "label": []}, "query": {"word": [], "label": []}, "types": []}

        # 构建类数据
        task_types = [] # N个类型
        randids = random.sample(range(types_num), N) # 每次随机选取N个类
        for randid in randids: # 每个随机类的索引
            ent_type = entity_types[randid]
            task_types.append(ent_type)
        each_task['types'] = task_types.copy() # 构建完每个任务中的类型信息

        def supque_dataset():
            # 开始构建support/query数据
            seq_list = []
            seqlabel_list = []
            for each_type_in_task in task_types: # 该任务中每个实体类型
                data_in_type = entity_type4sentslabels[each_type_in_task] # 原始数据中每个类型下的数据
                datanum_in_type = len(data_in_type) # 原始数据中每个类型数据的数量
                count = 0
                circulation_times = 0
                while count < K:
                    circulation_times += 1
                    select_data_ids = random.sample(range(datanum_in_type), K) # 从该类的原始数据中随机选择出K个
                    for select_data_id in select_data_ids:
                        each_data = data_in_type[select_data_id]
                        Sign = True
                        # 判断是不是所有的类型标签都在task_types中；如果都在，保留此条数据，count+1；如果不在，跳过此条数据
                        for token_label in each_data[1]:
                            if (token_label != 'O') and (token_label not in task_types): # 不合理的数据
                                Sign = False
                                break
                        if Sign and each_data[0] not in seq_list: # 所有的标签都在此次任务的实体类型中，才是合格的数据; 并且之前没有加入过
                            seq_list.append(each_data[0]) # word信息
                            seqlabel_list.append(each_data[1]) # 标签信息
                            count += 1
                            if count >= K:
                                break
                    if circulation_times == 10: # 防止死循环的发生，因为某类里面可能确实找不到K条满足条件的数据
                        # print("The count number is {}".format(count))
                        break

            return seq_list.copy(), seqlabel_list.copy()

        def judge_ins(sup_list, que_list):
            # 判断support和query会不会出现相同的数据
            ins_sign = False
            for tseq in que_list:
                if tseq in sup_list:
                    ins_sign = True
                    break
            return ins_sign

        support_list, support_label_list = supque_dataset()
        query_list, query_label_list = supque_dataset()

        while judge_ins(support_list, query_list): # 如果有交集，需要重新构建query
            query_list, query_label_list = supque_dataset()
            # print('CC in support and query intersection!')

        if len(support_list) > 0 and len(query_list) > 0:
            each_task['support']['word'] = support_list
            each_task['support']['label'] = support_label_list
            each_task['query']['word'] = query_list
            each_task['query']['label'] = query_label_list
            dataset.append(each_task)

        print("The {}-th task in {} is built!".format(data_iter, data_mode))

    return dataset.copy()


if __name__ == "__main__":
    root = 'D:/FEWAPTER4ESD/V1/'
    orig_alldata_path = root + 'APTNER.txt'
    new_ent_types_path = root + "APTNER_entity_types.json"

    all_data, all_types = load_file(orig_alldata_path, new_ent_types_path) # 获取原始数据

    # type_count(all_data, all_types)
    # exit()

    type4data = build_type4data(all_data, all_types) # 构建实体类别对应下的数据

    all_types = list(all_types)  # 所有的实体类
    all_types.remove("O")

    for N in [4, 6]:
        for K in [1, 3]:
            for data_sign in ['train', 'dev', 'test']:
                if data_sign == 'train':
                    data_num = 6000
                    ents_types = ['VULID', 'FILE', 'SECTEAM', 'LOC', 'TOOL', 'APT', 'ACT']
                elif data_sign == 'dev':
                    data_num = 3000
                    ents_types = ['SHA1', 'OS', 'URL', 'MAL', 'DOM', 'PROT', 'TIME']
                elif data_sign == 'test':
                    data_num = 3000
                    ents_types = ['ENCR', 'VULNAME', 'IP', 'MD5', 'SHA2', 'IDTY', 'EMAIL']
                else:
                    print('Error!')

                tasks = build_dataset(type4data, ents_types, 7, data_num, N, K, data_sign)
                print("There are {} tasks for {}-{}-{} built..., and we begin to write it into the file!".format(len(tasks), data_sign, N, K))

                wrt_path = root + data_sign + "_" + str(N) + '_' + str(K) + '.jsonl'
                with open(wrt_path, 'a+', encoding='utf-8') as wrt_file: # 文件对象
                    for task in tasks: # 每个任务
                        str_task = json.dumps(task) # 将每个任务转化成为字符串的形式
                        wrt_file.write(str_task+'\n') # 以字符串的形式逐行写入
