# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:58:54 2020

@author: TR
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


print('Reading excel File....')
data = pd.read_excel('Sample.xlsx','data')
#data = pd.read_excel('Book.xlsx', 'Sheet1')
df = pd.DataFrame(data)

print('Reading excel File Finished......')



colNum = len(df.columns)
rowNum = len(df.index)
print ('CSV file contain :\n' , 'rowsNumber :',rowNum, 'colNumber :',colNum)

print('**************************************\n')
print(df.info())

#print('first n Rows:\n',df.head(10))
print('\n--------------------------------------\n')


def WriteToExcel(xlsName,dfNew):
    print ('\n')
    print ("Writing to Excel File.........")
    with pd.ExcelWriter(xlsName) as writer:
        dfNew.to_excel(writer, sheet_name=xlsName)
    print ("Writing finished.........")



"""2"""
#WriteToExcel('LogisticHAM.xls',df)

y=df['PATIENT TYPE']
X = df.drop(['PATIENT TYPE'], axis=1)
print(y.value_counts())
y = np.where(df['PATIENT TYPE'] == 'Standing',0,1)

print ('-----------------------------------------\n')
print ('salary column y turns 0-1: ',len(y))
print(y)


"""3.-"""
df["CITY CODE"] = df["CITY CODE"].astype(str)




"""5.-"""
a = X.select_dtypes(include='object')  #normail satır satır aynı sütunları var
#print ('a:\n',a)
X = pd.concat([X, pd.get_dummies(a)], axis=1)
X = X.drop(['AGE', 'GENDER','CITY CODE','TIME ZONE',
        'ICD CODE'], axis=1)
print ('5.En Son Concat X:\n',X)


"""6.--"""
print ('\n')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix as cm
    
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=42)

print ('6.Train_X:',len(X_train),  '   Train_Y:',y_train)
print ('6.Test_X :', len(X_test),  '   Test_Y: '  ,y_test)


lr = LogisticRegression(random_state=42)
print ('\n')
lr.fit(X_train, y_train)


TestX = pd.DataFrame(X_test)
TestY = pd.DataFrame(y_test)
testVeri = pd.concat([TestX, TestY], axis=1)
WriteToExcel('testVeri.xlsx',testVeri)

"""7.-"""
predictions = lr.predict(X_test)   #predic. is an array for predict Test_x data
yy=np.array(y_test,dtype=int)
say=0
for i in range(len(predictions)):
    if predictions[i]==0 and yy[i]==0:
        say+=1
print ('  # of 0 corrects:',say)


score = round(accuracy_score(y_test, predictions), 3)  #corrects(0-1)/all test data=0.xx

cm1 = cm( y_test,predictions)

#print ('confusion matrix values: \n',cm1)
sns.heatmap(cm1, annot=True, fmt=".0f")
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Accuracy Score: {0}'.format(score), size = 15)
plt.show()


"""8.--"""
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions, target_names=['Standing', 'Inpatient']))


"""9.--"""
plt.figure(figsize=(8,8))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

roc_auc = roc_auc_score(y_test, lr.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, lr.predict_proba(X_test)[:,1])

plt.plot(fpr, tpr, label='LR (auc = %0.3f)' % roc_auc, color='navy')
plt.plot([0, 1], [0, 1],'r--')
plt.title('ROC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()


"""10.-"""
from sklearn.feature_selection import RFE

rfe = RFE(lr, 10)

rfe = rfe.fit(X_train, y_train)
print(rfe.ranking_)
X_train[X_train.columns[rfe.ranking_==1].values].head()


"""11.--"""
import statsmodels.api as sm
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)


logit_model=sm.Logit(y_train, X_train[X_train.columns[rfe.ranking_==1].values])

result=logit_model.fit()

predictions= result.predict(X_test[X_test.columns[rfe.ranking_==1].values])
print(classification_report(y_test, predictions.round(), target_names=['Standing', 'Inpatient']))
print(result.summary([df.columns]))
print (result.conf_int)






