# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 10:58:08 2019

@author: 91603
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 09:52:50 2019

@author: 91603
"""

#数据处理包导入
import numpy as np
import pandas as pd
#画图包导入
import matplotlib.pyplot as plt
#使用自带的样式进行美化
plt.style.use(style="ggplot")
import missingno as msno
import seaborn as sns
#应用默认的seaborn主题、缩放和调色板
sns.set()

#读取数据
train_names=["date",         #销售日期
             "price",        #销售价格
             "bedrooms",     #卧室数
             "bathrooms",    #浴室数
             "sqft_living",  #房屋面积
             "sqft_lot",     #停车场面积
             "floors",       #楼层数
             "grade",        #房屋评分（数据来源于King County房屋评分系统）
             "sqft_above",   #地上建筑面积
             "sqft_basement",#地下室面积
             "yr_built",     #建成年份
             "yr_renovated", #修复年份
             "lat",          #经度
             "long"]         #纬度
train=pd.read_csv("kc_house_data.csv",names=train_names)

#数据处理(数据格式转换)
train=pd.read_csv("kc_house_data.csv",names=train_names,parse_dates=["date","yr_built","yr_renovated"])

#房价分布曲线
plt.figure(figsize=(10,5))#设置画布尺寸
print("skew:",train.price.skew())#输出偏度
sns.distplot(train['price'])#打印房价分布曲线

#通过对数变换改变数据的线性度
#通过变换，分布更接近正态分布
target=np.log(train.price)#返回自然对数
plt.figure(figsize=(10,5))#设置画布尺寸
sns.distplot(target)#打印已优化的房价分布曲线

#检测数值特征和目标变量之间的相关性
#输出下列项目参数之间的相关性
corrMat=train[['price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','grade','sqft_above','sqft_basement','yr_built','yr_renovated','lat','long']].corr()
mask=np.array(corrMat)#数组切片
#生成相关系数的下三角系数图像
mask[np.tril_indices_from(mask)]=False
plt.subplots(figsize=(20,10))
plt.xticks(rotation=60)#设置刻度标签角度
sns.heatmap(corrMat,mask=mask,vmax=.8,square=True,annot=True)
print(corrMat["price"].sort_values(ascending=False))#输出价格与各项目之间的相关系数

#构建房价预测模型
features=['sqft_living',
          'grade',
          'sqft_above',
          'bathrooms',
          'sqft_basement',
          'bedrooms',
          'lat',
          'floors',
          'yr_renovated',
          'sqft_lot',
        ]
x=train[features]
y=np.log(train['price'])

#特征预处理
from sklearn.preprocessing import MinMaxScaler
x_copy = x[:]
scaler = MinMaxScaler()
x_transformed = scaler.fit_transform(x_copy)

#生成训练集和测试集
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_transformed,y,random_state=0,test_size=0.5)

#建立模型
from sklearn import linear_model
lm = linear_model.LinearRegression()
model = lm.fit(x_train,y_train)
print(model.intercept_,model.coef_)#输出截距(权重)和斜率

#评估模型
model.score(x_test,y_test)
print ('R^2 is: \n', model.score(x_test,y_test))

#生成预测
test = pd.read_csv("kc_house_data.csv", names=train_names ,parse_dates=["date","yr_built"])
features = ['sqft_living',
            'grade',
            'sqft_above',
            'bathrooms',
            'sqft_basement',
            'bedrooms',
            'long',
            'floors',
            "yr_renovated",
            "sqft_lot"
           ]
x2 = test[features]
scaler_test = scaler.fit_transform(test[features])
final_predicts = np.exp(model.predict(scaler_test))
np.savetxt('price.csv', final_predicts, delimiter = ',')  