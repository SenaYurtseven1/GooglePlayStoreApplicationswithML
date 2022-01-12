from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('./clean_googleplaystore_dataset.csv')

df.drop('Unnamed: 0',inplace=True,axis=1)
df.drop('Size',inplace=True,axis=1)

#df = df.sample(n=5000 ,replace="False")

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


onehotencodeddata = pd.get_dummies(df, columns = ['Category','Content Rating','AppRating','Type','month','Minimum Android'])

#Split
features = onehotencodeddata.drop('Installs',axis = 1)
target = onehotencodeddata['Installs']
class_names = target.unique()
class_names=class_names[class_names !='0']

features_train, features_test, target_train, target_test = train_test_split(features, target,test_size=0.2 ,random_state=1)

####Random Forest

'''
param_grid = {'max_depth': [10, 20, 30, 40, 50,60,70,80,90],
              'criterion':['gini','entropy'],
              'n_estimators':[20,30,40,50,60,70,80,90,100,110,120,130]
              }

grid = GridSearchCV(RandomForestClassifier(random_state=0), param_grid=param_grid, refit = True, verbose = 3,scoring="accuracy")
grid.fit(features_train, target_train)

# print best parameter after tuning
print(grid.best_params_)
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)
grid_predictions = grid.predict(features_test)
'''

# Create random forest classifier object
randomforest=RandomForestClassifier(random_state=0, n_jobs=-1,max_depth=20,n_estimators=30)
# Train model
model = randomforest.fit(features_train, target_train)

# evaluate the model
cv = KFold(n_splits=5, shuffle=True, random_state=1)

cv_results = cross_val_score(model,features_test, target_test, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

# report performance
print('Accuracy of Random Forest Tree classifier with split on test set(max depth set): {:.2f}' .format(model.score(features_test, target_test)))

target_predicted = model.predict(features_test)
print('Predicted Class: %s' % target_predicted[0])

# Create confusion matrix
matrix = confusion_matrix(target_test, target_predicted)
# Create pandas dataframe
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)

# Create a classification report
print(classification_report(target_test,target_predicted,target_names=class_names))

