import numpy as np 
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold,GridSearchCV, learning_curve, train_test_split, cross_val_score
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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

'''
####Training Set and Test Set Depth Performance values
decisiontreetainscore = []
decisiontreetestscore = []
maxdepth = []


for i in range(20,40):
    maxdepth.append( i)   
    clf = DecisionTreeClassifier(random_state = 0,max_depth=i).fit(features_train, target_train)
    decisiontreetainscore.append(clf.score(features_train, target_train))
    decisiontreetestscore.append(clf.score(features_test, target_test))
    print(i)

clf = DecisionTreeClassifier(random_state = 0,max_depth=20).fit(features_train, target_train)
plt.plot(maxdepth, decisiontreetainscore, color='r', label='Train Score')
plt.plot(maxdepth, decisiontreetestscore, color='g', label='Test Score')

plt.xlabel("Depth Size")
plt.ylabel("Train Performance")
plt.title("Train Performance by Depth Size")
'''

'''
###Grid Search
param_grid = {'max_depth': [10, 20, 30, 40, 50,60,70,80,90],
              'criterion':['gini','entropy']}

grid = GridSearchCV(DecisionTreeClassifier(random_state=1), param_grid=param_grid, refit = True, verbose = 3,scoring="accuracy")
grid.fit(features_train, target_train)

# print best parameter after tuning
print(grid.best_params_)
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)
grid_predictions = grid.predict(features_test)
'''

'''
###Create a learning curve
train_sizes, train_scores, test_scores = learning_curve( DecisionTreeClassifier(), features, target, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace( 0.01, 1.0, 50))
# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
# Draw lines
plt.plot(train_sizes, train_mean, '--', color="#111111", label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")
# Draw bands
plt.fill_between(train_sizes, train_mean - train_std,
train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std,
test_mean + test_std, color="#DDDDDD")
# Create plot
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"),
plt.legend(loc="best")
plt.tight_layout()
plt.show()
'''

clf = DecisionTreeClassifier(random_state = 0,max_depth=10,criterion='entropy').fit(features_train, target_train)
#print('Accuracy of Decision Tree classifier on training set(max depth): {:.2f}'.format(clf.score(features_train, target_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}' .format(clf.score(features_test, target_test)))

cv = KFold(n_splits=5, shuffle=True, random_state=1)
cv_results_max_depth = cross_val_score(clf,features_test, target_test, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print("Accuracy of Decision Tree classifier with n-fold on training set:", format(100 * cv_results_max_depth.mean(), ".2f") + "%")

target_predicted = clf.predict(features_test)
print('Predicted Class: %s' % target_predicted[0])
# Create a classification report
print(classification_report(target_test,target_predicted,target_names=class_names))


# Create confusion matrix
matrix = confusion_matrix(target_test, target_predicted)
# Create pandas dataframe
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
# Create heatmap
sns.heatmap(dataframe,annot=True, cbar=None, cmap="Blues")
plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()

