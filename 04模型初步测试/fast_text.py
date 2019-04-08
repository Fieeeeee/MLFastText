# coding=utf-8
import fasttext
import datetime
import pandas as pd
from sklearn.utils import shuffle

start_time = datetime.datetime.now()
print('程序开始')

data = pd.read_csv('../data/train_added_label.txt', sep='\t', encoding='utf-8', header=0)
data = shuffle(data)
data.to_csv('../data/train_suffled.txt',sep='\t',index=False)

classifier = fasttext.supervised('../data/train_suffled.txt', '../model/classifier.model',loss='hs',word_ngrams=2,bucket= 2000000,lr=0.5)
pre_time = datetime.datetime.now()
print('预测结束，耗时：', (pre_time - start_time).seconds, 's')

result = classifier.test('../data/test_suffled.txt')
test_time = datetime.datetime.now()
print('测试结束，耗时：', (test_time - pre_time).seconds, 's')

print("准确率:", result.precision)

end_time = datetime.datetime.now()
print('程序结束,共耗时：', (end_time - start_time).seconds, 's')
