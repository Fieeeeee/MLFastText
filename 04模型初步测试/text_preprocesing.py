import jieba
import csv
import datetime
import itertools
import re
import logging
import pandas as pd
from sklearn.utils import shuffle

start_time = datetime.datetime.now()


def savefile(savepath, content):
    fp = open(savepath, "w", encoding='utf-8', errors='ignore')
    fp.write(content)
    fp.close()


print('----数据预处理开始')

# 读取数据集并划分
csv.register_dialect('mydialect', delimiter='\t', quoting=csv.QUOTE_ALL)
train_item = open('../data/train_item.tsv', 'w', encoding='utf-8')
train_type = open('../data/train_type.tsv', 'w', encoding='utf-8')

with open('../data/train.tsv', encoding='utf-8') as csvfile:
    file_list = csv.reader(csvfile, 'mydialect')
    for line in file_list:
        train_item.write(line[0] + '\n')
        train_type.write(line[1] + '\n')
csv.unregister_dialect('mydialect')
print('----数据集初始化结束')

# 分词 cut_for_search
jieba.setLogLevel(logging.INFO)
with open('../data/train_item.tsv', encoding='utf-8') as train_items:
    seged_train_item = [
        re.sub('[a-zA-Z0-9’!"#$%&\'()（）*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+', '', ' '.join(jieba.cut(r))) for r in
        train_items.readlines()]
with open('../data/test.tsv', encoding='utf-8') as train_items:
    seged_test = [
        re.sub('[a-zA-Z0-9’!"#$%&\'()（）*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+', '', ' '.join(jieba.cut(r))) for r in
        train_items.readlines()]
savefile('../data/seged_train_item.tsv', ''.join(seged_train_item))
savefile('../data/seged_test.tsv', ''.join(seged_test))
print('----分词去停用词结束')

with open("../data/train_type.tsv", encoding='utf-8') as f:
    train_type = [r.rstrip('\n').replace('\t', '') for r in f.readlines()]
with open("../data/seged_train_item.tsv", encoding='utf-8') as f:
    seged_train_item = [r.rstrip('\n').replace('\t', '') for r in f.readlines()]
result = itertools.zip_longest(seged_train_item, train_type, fillvalue=' ')
with open("../data/train_added_label.txt", "w", encoding='utf-8') as f:
    [f.write('\t__label__'.join(r) + "\n") for r in result]
print('----合并训练集结束')

data = pd.read_csv('../data/train_added_label.txt', sep='\t', encoding='utf-8', header=0)
data = shuffle(data)
train_shuffled = data.head(450000)
test_shuffled = data.tail(49946)
train_shuffled.to_csv('../data/train_suffled.txt', sep='\t', index=False)
test_shuffled.to_csv('../data/test_suffled.txt', sep='\t', index=False)
end_time = datetime.datetime.now()
print('数据预处理结束，总耗时：', (end_time - start_time).seconds / 60, 'min')
