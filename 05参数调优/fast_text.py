import jieba
import csv
import datetime
import itertools
import pandas as pd
from sklearn.utils import shuffle
import fasttext

# txt_pre_start = datetime.datetime.now()
# with open("../data/train_type.tsv", encoding='utf-8') as f:
#     train_type = [r.rstrip('\n').replace('\t', '') for r in f.readlines()]
# with open("../data/seged_train_item.tsv", encoding='utf-8') as f:
#     seged_train_item = [r.rstrip('\n').replace('\t', '') for r in f.readlines()]
# result = itertools.zip_longest(seged_train_item, train_type, fillvalue=' ')
# with open("../data/train_added_label.txt", "w", encoding='utf-8') as f:
#     [f.write('\t__label__'.join(r) + "\n") for r in result]
data = pd.read_csv('../data/train_added_label.txt', sep='\t', encoding='utf-8', header=0)
data = shuffle(data)
# txt_pre_end = datetime.datetime.now()
# print('文本预处理结束,耗时:', (txt_pre_end - txt_pre_start).seconds, 's')

data.to_csv('../data/train_suffled.txt', sep='\t', index=False)
start_time = datetime.datetime.now()
# classifier = fasttext.supervised('../data/train_suffled.txt', 'classifier.model')
# classifier = fasttext.supervised('../data/train_suffled.txt', 'classifier.model',loss='ns')
# classifier = fasttext.supervised('../data/train_suffled.txt', 'classifier.model',loss='hs')

# classifier = fasttext.supervised('../data/train_suffled.txt', 'classifier.model',loss='hs',lr=0.6)
# classifier = fasttext.supervised('../data/train_suffled.txt', 'classifier.model',loss='hs',dim=25)
# classifier = fasttext.supervised('../data/train_suffled.txt', 'classifier.model',loss='hs',ws=10)
# classifier = fasttext.supervised('../data/train_suffled.txt', 'classifier.model',loss='hs',epoch=15)
# classifier = fasttext.supervised('../data/train_suffled.txt', 'classifier.model',loss='hs',min_count=8)
# classifier = fasttext.supervised('../data/train_suffled.txt', 'classifier.model',loss='hs',neg=5)
classifier = fasttext.supervised('../data/train_suffled.txt', 'classifier.model',loss='hs',word_ngrams=1,bucket=2000000,lr=0.5)






pre_time = datetime.datetime.now()
print('预测结束，耗时：', (pre_time - start_time).seconds, 's')

result = classifier.test('../data/train_suffled.txt')
test_time = datetime.datetime.now()
print('测试结束，耗时：', (test_time - pre_time).seconds, 's')
print("准确率:", result.precision)
#
#
# end_time = datetime.datetime.now()
# print('程序结束,共耗时：', (end_time - start_time).seconds, 's')
