import jieba
import csv
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def savefile(savepath, content):
    fp = open(savepath, "w", encoding='utf-8', errors='ignore')
    fp.write(content)
    fp.close()


def readfile(path):
    fp = open(path, "r", encoding='utf-8', errors='ignore')
    content = fp.read()
    fp.close()
    return content


# 读取数据集
train_item_set = readfile('../data/train_item.tsv')
print('----读取文件结束----')

# 分词
start_time = datetime.datetime.now()
jieba.add_word('ITEM_NAME')
# 全模式
# seged_train_item = ' '.join(jieba.cut(train_item_set,cut_all=True))
# 默认模式/精确模式
# seged_train_item = ' '.join(jieba.cut(train_item_set))
# 搜索模式
seged_train_item = ' '.join(jieba.cut_for_search(train_item_set))
end_time = datetime.datetime.now()
print('预测结束，分词耗时：', (end_time - start_time).seconds, 's')

print('----分词结束----')

savefile('../data/seged_train_item.tsv', seged_train_item)
print('----保存数据集----')
# 读取文件，并转换为列表
train_item = pd.read_csv('../data/seged_train_item.tsv', sep='\t', encoding='utf-8', header=0)
train_item_list = train_item['ITEM_NAME '].tolist()
train_type = pd.read_csv('../data/train_type.tsv', sep='\t', encoding='utf-8', header=0)
train_type_list = train_type['TYPE'].tolist()
print('读取文件结束')

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(train_item_list, train_type_list, test_size=0.25,
                                                    random_state=12345)
print('训练集数据条数：', len(x_train), '\n', '测试集数据条数', len(x_test))
print('划分数据集结束')

# # 构建特征矩阵
cv = CountVectorizer()
x_train = cv.fit_transform(x_train).toarray()
x_test = cv.transform(x_test).toarray()
print('构建特征矩阵结束')

# 朴素贝叶斯
nb = MultinomialNB(alpha=1.0)

nb.fit(x_train, y_train)
y_predict = nb.predict(x_test)

print('准确率为：', '{:.2%}'.format(nb.score(x_test, y_test)))
print('程序结束')
