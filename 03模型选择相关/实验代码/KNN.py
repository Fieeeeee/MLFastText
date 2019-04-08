import pandas as pd
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

print('程序开始')

# 读取文件，并转换为列表
train_item = pd.read_csv('../data/seged_train_item.tsv', sep='\t', encoding='utf-8', header=0)
train_item_list = train_item['ITEM_NAME '].tolist()
train_type = pd.read_csv('../data/train_type.tsv', sep='\t', encoding='utf-8', header=0)
train_type_list = train_type['TYPE'].tolist()
# test_item = pd.read_csv('../data/seged_test.tsv', sep='\t', encoding='utf-8', header=0)
# test_item_list= train_item['ITEM_NAME '].tolist()
print('读取文件结束')

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(train_item_list, train_type_list, test_size=0.25, random_state=12345)
# print(type(x_test))
print('训练集数据条数：',len(x_train),'\n','测试集数据条数',len(x_test))
print('划分数据集结束')

# # 构建特征矩阵
cv = CountVectorizer()
x_train = cv.fit_transform(x_train).toarray()
x_test = cv.transform(x_test).toarray()
print('构建特征矩阵结束')

# tf = TfidfVectorizer()
# x_train = tf.fit_transform(x_train)
# x_test = tf.transform(x_test)
# print('特征',tf.get_feature_names())
# print('构建特征矩阵结束')
#
# 数据预处理
# 标准化
# std = StandardScaler()
# x_train = std.fit_transform(np.array(x_train))
# x_test = std.transform(np.array(x_test))
# print('数据标准化结束')
# 主成分分析
# pca = PCA(n_components=0.9)
# x_train = pca.fit_transform(np.array(x_train))
# x_test = pca.transform(np.array(x_test))
# print('数据预处理结束')

# 朴素贝叶斯
# nb = MultinomialNB(alpha=1.0)
# nb = GaussianNB()
nb =KNeighborsClassifier()
start_time = datetime.datetime.now()
nb.fit(x_train, y_train)
y_predict = nb.predict(x_test)
end_time = datetime.datetime.now()
print('预测结束，耗时：',(end_time-start_time).seconds,'s')
print('准确率为：', '{:.2%}'.format(nb.score(x_test, y_test)))
# print('每个类别的精确率和召回率为：',classification_report(y_test,y_predict,train_type_list))
print('程序结束')
