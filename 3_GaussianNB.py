import numpy as np

## GaussianNB预测过程
#定义两个向量
X=np.array([[-1,-1],[-2,-1],[-3,-2],[1,1],[2,1],[3,1]])
Y=np.array([1,1,1,2,2,2])

#定义模型
from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()

#拟合数据（训练模型）
clf.fit(X,Y)

#预测能力
#看predict的结果
print('---Predict result by predict---')
print(clf.predict([[-0.8,-1]]))

#看predict_proba的结果
print('---Predict result by predict_proba---')
print(clf.predict_proba([[-0.8,-1]]))

#看predict_log_proba的结果
print('---Predict result by predict_log_proba---')
print(clf.predict_log_proba([[-0.8,-1]]))