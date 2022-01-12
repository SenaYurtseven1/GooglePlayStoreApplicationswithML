from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('./clean_googleplaystore_dataset.csv')
df.drop('Unnamed: 0',inplace=True,axis=1)
df = df.sample(n=10000,replace="False")
#df['Installs']  = df['Installs'].astype(str)

#Minimize the Install target attribute values
Install=[]
#Target Feature minimize
for i in df['Installs']:
  if i <=500:
    Install.append('Low')
  elif i>500 and i<=100000:
      Install.append('Medium')
  elif i>100000:
      Install.append('High')
  else:
      Install.append('Unranked')
      
df.drop('Installs',inplace=True,axis=1)    
df['Installs']=Install

onehotencodeddata = pd.get_dummies(df, columns = ['Category','Content Rating','AppRating','Type','month'])
features = onehotencodeddata.drop('Installs',axis = 1)
target = onehotencodeddata.Installs


###Split
features_train, features_test, target_train, target_test = train_test_split(features, target,test_size=0.2 ,random_state=1)
class_names = target.unique()



"""
# defining parameter range auto= 1/n_features
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100,'auto'],
              'kernel': ['linear']}

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3,scoring="accuracy")

grid.fit(features_train, target_train)

# print best parameter after tuning
print(grid.best_params_)
 
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)


grid_predictions = grid.predict(features_test)

print(grid_predictions)

"""

svm = SVC(C=0.1,kernel='linear')
svm.fit(features_train, target_train)
print('Accuracy of SVM classifier on test set: {:.2f}'.format(svm.score(features_test, target_test)))

target_predicted = svm.predict(features_test)
# print classification report
print(classification_report(target_test, target_predicted))

