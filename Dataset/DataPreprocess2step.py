import json

setting = 'dev_4_3'
filepath = 'D:/FEWAPTER4ESD/V1/'+setting+'.jsonl'
wfilepath = 'D:/FEWAPTER4ESD/V2/'+setting+'.jsonl'

datas = open(filepath).readlines()
datas = [json.loads(x.strip()) for x in datas]
data_num = 0

with open(wfilepath, 'a+', encoding='utf-8') as wrt_file:
    for d in datas:
        types = d['types']
        slabels = []
        qlabels = []
        stype_ll = d['support']['label']
        qtype_ll = d['query']['label']
        for lw in stype_ll:
            for li in lw:
                slabels.append(li)

        for lw in qtype_ll:
            for li in lw:
                qlabels.append(li)

        sslabel_ = set(slabels)
        qqlabel_ = set(qlabels)
        # if len(sslabel_) != len(types)+1 or len(qqlabel_) != len(types)+1:
        #     print(d)
        #     print(sslabel_)
        #     print(qqlabel_)
        if sslabel_ == qqlabel_ or qqlabel_.issubset(sslabel_):
            str_task = json.dumps(d)
            wrt_file.write(str_task + '\n')
            data_num += 1
        if data_num == 800:
            break

print(data_num)
