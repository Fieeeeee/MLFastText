import jieba
import csv
import datetime
import itertools
import re
import logging
import pandas as pd
from sklearn.utils import shuffle
import fasttext







# print('----数据预处理开始')
#
# # 读取数据集并划分
# csv.register_dialect('mydialect', delimiter='\t', quoting=csv.QUOTE_ALL)
# train_item = open('../data/train_item.tsv', 'w', encoding='utf-8')
# train_type = open('../data/train_type.tsv', 'w', encoding='utf-8')
#
# with open('../data/train.tsv', encoding='utf-8') as csvfile:
#     file_list = csv.reader(csvfile, 'mydialect')
#     for line in file_list:
#         train_item.write(line[0] + '\n')
#         train_type.write(line[1] + '\n')
# csv.unregister_dialect('mydialect')
# print('----数据集初始化结束')


with open("../data/train_type.tsv", encoding='utf-8') as f:
    train_type = [r.rstrip('\n').replace('\t', '') for r in f.readlines()]
with open("../data/seged_train_item.tsv", encoding='utf-8') as f:
    seged_train_item = [r.rstrip('\n').replace('\t', '') for r in f.readlines()]
result = itertools.zip_longest(seged_train_item, train_type, fillvalue=' ')
with open("../data/train_added_label.txt", "w", encoding='utf-8') as f:
    [f.write('\t__label__'.join(r) + "\n") for r in result]
print('数据预处理结束')

data = pd.read_csv('../data/train_added_label.txt', sep='\t', encoding='utf-8', header=0)
data = shuffle(data)
train_shuffled = data.head(15000)
test_shuffled = data.tail(4999)

train_shuffled.to_csv('../data/train_suffled.txt', sep='\t', index=False)
test_shuffled.to_csv('../data/test_suffled.txt', sep='\t', index=False)



data = pd.read_csv('../data/train_added_label.txt', sep='\t', encoding='utf-8', header=0)
data = shuffle(data)
data.to_csv('../data/train_suffled.txt',sep='\t',index=False)
start_time = datetime.datetime.now()
classifier = fasttext.supervised('../data/train_suffled.txt', 'classifier.model',loss='hs',word_ngrams=2,bucket= 2000000,lr=0.5)
pre_time = datetime.datetime.now()
print('预测结束，耗时：', (pre_time - start_time).seconds, 's')

result = classifier.test('../data/test_suffled.txt')
test_time = datetime.datetime.now()
print('测试结束，耗时：', (test_time - pre_time).seconds, 's')
print("准确率:", result.precision)
# print("召回率:", result.recall)

# end_time = datetime.datetime.now()
# print('程序结束,共耗时：', (end_time - start_time).seconds, 's')

