from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model

def log_mean_absolute_error(y,y_pred):
    temp = len(y)*[1]
    y = np.array(y) + np.array(temp)
    y_pred = np.array(y_pred) + np.array(temp)
    return mean_absolute_error(np.log(y),np.log(y_pred))

def score(tail_train,pre_train):
    score = 0
    for col in tail_train.columns:
        if col == 'vid':
            pass
        else:
            score += log_mean_absolute_error(tail_train[col],pre_train[col])
    return score/5

train = pd.read_csv('./data/train_set.csv',encoding='utf-8')
test = pd.read_csv('./data/test_set.csv',encoding='utf-8')

# 修改列名 收缩压,舒张压,血清甘油三酯,血清高密度脂蛋白,血清低密度脂蛋白
train.rename(columns={u'收缩压':'A', u'舒张压':'B', u'血清甘油三酯':'C',u'血清高密度脂蛋白':'D',u'血清低密度脂蛋白':'E'}, inplace = True)
test.rename(columns={u'收缩压':'A', u'舒张压':'B', u'血清甘油三酯':'C',u'血清高密度脂蛋白':'D',u'血清低密度脂蛋白':'E'}, inplace = True)
#print(train.hea)

num_cols = ['vid']
for col in train.columns:
    if train[col].dtype != 'object':
        num_cols.append(col)

all_data = pd.concat([train,test])
all_data = all_data[num_cols]

#缺失值处理
for col in all_data.columns:
    if col != 'vid':
        all_data[col] = pd.to_numeric(all_data[col],errors='coerce')
        all_data[col].fillna(all_data[col].mean(),inplace = True)

#print(all_data.describe())

train_data=all_data[all_data['vid'].isin(train['vid'])]
test_data=all_data[all_data['vid'].isin(test['vid'])]

# 测试
head_train = train_data.iloc[:20000,:]
tail_train = train_data.iloc[20000:,:]

#clf = linear_model.MultiTaskLasso(alpha=0.1)
clf = linear_model.MultiTaskElasticNet(alpha=0.1)
X = head_train[num_cols[6:]]
y = head_train[num_cols[1:6]]

clf.fit(X,y)

test_X = tail_train[num_cols[6:]]

tail_Y = tail_train[num_cols[1:6]]
y_pre = clf.predict(test_X)
result = pd.DataFrame({'vid':tail_train.vid,'A':y_pre[:,0],'B':y_pre[:,1],'C':y_pre[:,2],'D':y_pre[:,3],'E':y_pre[:,4]},columns=['vid','A','B','C','D','E'])
print(score(tail_Y,result))


# clf = linear_model.MultiTaskLasso(alpha=0.1)
# X = train_data[num_cols[6:]]
# y = train_data[num_cols[1:6]]
# #print(y.columns)

# clf.fit(X,y)

# test_X = test_data[num_cols[6:]]
# y_pre = clf.predict(test_X)

# result = pd.DataFrame({'vid':test_data.vid,'A':y_pre[:,0],'B':y_pre[:,1],'C':y_pre[:,2],'D':y_pre[:,3],'E':y_pre[:,4]},columns=['vid','A','B','C','D','E'])
# #print(select_train.describe())
# #print(result.head())

# result.round({'A':3,'B':3,'C':3,'D':3,'E':3}).to_csv('result0427.csv',index=None,header=None)
#vectorizer = CountVectorizer()
#X = vectorizer.fit_transform(corpus)

train = pd.read_csv("./data/meinian_round1_train_20180408.csv",encoding='gbk')
print('train',len(train.vid))
test = pd.read_csv("./data/meinian_round1_test_a_20180409.csv",encoding='gbk')
print('test',len(test.vid))

columns = ['vid','A','B','C','D','E']
train.columns = columns
test.columns = columns
# #print(train.head())

# 转换成数字
train.A = pd.to_numeric(train.A,errors='coerce')
train.B = pd.to_numeric(train.B,errors='coerce')
train.C = pd.to_numeric(train.C,errors='coerce')

# # 删除缺失值
train = train.dropna(axis=0,how='any')

print('train',len(train.vid))
#划分数据
head_train = train.iloc[:20000,:]
tail_train = train.iloc[20000:,:]
#print(head_train.describe())
#print(tail_train.describe())
pre_train = tail_train.copy()
pre_train.A = head_train.A.mean()
pre_train.B = head_train.B.mean()
pre_train.C = head_train.C.mean()
pre_train.D = head_train.D.mean()
pre_train.E = head_train.E.mean()

# test_pre = test.copy()
# for col in train.columns:
#     if col == 'vid':
#         pass
#     else:
#         test_pre[col] = round(train[col].mean(), 3)

# test_pre.to_csv('result.csv',index=None,header=None)

print(score(tail_train,pre_train))
#train.A.mean()
#plt.show()

## 均值提交线上结果0.0503
