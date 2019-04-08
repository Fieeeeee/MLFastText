import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('../data_old/train.tsv',sep='\t',encoding='utf-8')
test = pd.read_csv('../data_old/test.tsv',sep='\t',encoding='utf-8')
train_type = train['TYPE']
#查看数据集数量
# print(len(train_type))
print(len(test))

# 查看有多少种类
# set=set()
# for i in train_type:
#     set.add(i)
# print(len(set))

# c = Counter(train_type)
# print(c)

# 数据分布图
# d = train.groupby(train['TYPE']).count()
# img = d.plot()
# plt.show()
# img.get_figure().savefig('fig.png')

