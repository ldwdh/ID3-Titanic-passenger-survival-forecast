import pandas as pd
train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')
#数据清洗
train_data['Age'].fillna(train_data['Age'].mean(),inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)
train_data['Fare'].fillna(train_data['Fare'].mean(),inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)
train_data['Embarked'].fillna('S',inplace=True)
test_data['Embarked'].fillna('S',inplace=True)
#特征选择
features=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
train_features=train_data[features]
train_labels=train_data['Survived']
test_features=test_data[features]
#字符串转数值
from sklearn.feature_extraction import DictVectorizer
dvec=DictVectorizer(sparse=False)
train_features=dvec.fit_transform(train_features.to_dict(orient='record'))
#构造ID3决策树
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(criterion='entropy')
#决策树训练
clf.fit(train_features,train_labels)
#决策树预测
test_features=dvec.fit_transform(test_features.to_dict(orient='record'))
pred_lables=clf.predict(test_features)
#k折交叉验证
import numpy as np
from sklearn.model_selection import cross_val_score
accuracy=np.mean(cross_val_score(clf,train_features,train_labels,cv=10))
print('决策树准确率为{:.2f}'.format(accuracy))

