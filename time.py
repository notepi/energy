#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 17:46:58 2019

@author: pan
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 16:28:46 2018

@author: pan
"""
from time import time
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
import math
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import precision_recall_fscore_support



if __name__ == "__main__":
    data = pd.read_csv(u'能源消费.csv',encoding = "GBK" )
    
    length=3
    timeSequence=[]
    for i in range(len(data)):
        if i > (len(data)-length):
            break
            pass
        timeSequence.append(pd.DataFrame(data.iloc[i:i+length,-1].values).T)
        pass
    timeSequence=pd.concat(timeSequence,axis=0)
    timeSequence=timeSequence.div(10000)
    
    # 生成训练数据和测试数据
    XdataTrain = timeSequence.iloc[:-3, :-1]
    TagTrain = timeSequence.iloc[:-3, -1]

    XdataTest = timeSequence.iloc[-3:, :-1]
    TagTest = timeSequence.iloc[-3:, -1]

    ####============================================================================
    print("=================train model")
    # 使用SVM作为分类器
    clf = svm.SVR()

    # use a full grid over all parameters
    param_grid = {"C": np.logspace(-2, 2, 10),
                  "gamma": np.logspace(-2, 2, 10)
                  }

    # 网格搜索， grid search
    grid_search = GridSearchCV(clf, param_grid=param_grid,cv=2)
    start = time()
    grid_search.fit(XdataTrain, TagTrain)

    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.cv_results_['params'])))
    svr = grid_search
    y_hat=svr.predict(XdataTest)
    
    
    
    mse = np.average((y_hat - np.array(TagTest)) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    print ("Mean Squared Error is:",mse)
    print ("Root Mean Squared Error is:",rmse)

    t = np.arange(len(XdataTest))
    plt.plot(t, TagTest, 'r-', linewidth=2, label='Test')
    plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
    
    pass
