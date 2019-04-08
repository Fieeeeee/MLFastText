import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
print('程序开始')

# 读取文件，并转换为列表
train_item = pd.read_csv('../data/seged_train_item.tsv', sep='\t', encoding='utf-8', header=0)
train_item_list = train_item['ITEM_NAME '].tolist()
train_type = pd.read_csv('../data/train_type.tsv', sep='\t', encoding='utf-8', header=0)
train_type_list = train_type['TYPE'].tolist()
print('读取文件结束')

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(train_item_list, train_type_list, test_size=0.25, random_state=12345)
print('划分数据集结束')

# # 构建特征矩阵
cv = CountVectorizer()
x_train = cv.fit_transform(x_train).toarray()
x_test = cv.transform(x_test).toarray()
print('构建特征矩阵结束')


nb = RandomForestClassifier()
start_time = datetime.datetime.now()
nb.fit(x_train, y_train)
y_predict = nb.predict(x_test)
end_time = datetime.datetime.now()
print('预测结束，耗时：',(end_time-start_time).seconds,'s')
print('准确率为：', '{:.2%}\n'.format(nb.score(x_test, y_test)))
# print('每个类别的精确率和召回率为：',classification_report(y_test,y_predict,train_type_list))
print('程序结束')
