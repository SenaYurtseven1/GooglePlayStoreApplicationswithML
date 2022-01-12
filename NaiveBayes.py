import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import BernoulliNB 
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
#Discrete sayılar için çalıştırılıyor örneğin bir döküman içerisinde word sayısı gibi
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import CategoricalNB

df = pd.read_csv('./clean_googleplaystore_dataset.csv')
df.drop('Unnamed: 0',inplace=True,axis=1)
#df = df.sample(n=500000,replace="False")
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


#print(df['Installs'].value_counts())
onehotencodeddata = pd.get_dummies(df, columns = ['Category','Content Rating','AppRating','Type','month'])
features = onehotencodeddata.drop('Installs',axis = 1)
target = onehotencodeddata.Installs
class_names = target.unique()

"""Split"""
features_train, features_test, target_train, target_test = train_test_split(features, target,test_size=0.2 ,random_state=1)

#Gaussian
gnb = GaussianNB()
gnb.fit(features_train, target_train)
print('Accuracy of GNB classifier on training set: {:.2f}'.format(gnb.score(features_train, target_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'.format(gnb.score(features_test, target_test)))

#######N-fold Cross Validation
cv = KFold(n_splits=5, shuffle=True, random_state=1)
cv_results = cross_val_score(gnb,features_test, target_test, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print("Accuracy of Bernoulli Naive Bayes classifier with n-fold on test set:", format(100 * cv_results.mean(), ".2f") + "%")

bnb = BernoulliNB()
bnb.fit(features_train, target_train)
print('Accuracy of Bernolli classifier on training set: {:.2f}'.format(bnb.score(features_train, target_train)))
print('Accuracy of Bernolli classifier on test set: {:.2f}'.format(bnb.score(features_test, target_test)))

cv = KFold(n_splits=5, shuffle=True, random_state=1)
cv_results = cross_val_score(bnb,features_test, target_test, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print("Accuracy of Bernoulli Naive Bayes classifier with n-fold on test set:", format(100 * cv_results.mean(), ".2f") + "%")

target_predicted = gnb.predict(features_test)
print('Predicted Class: %s' % target_predicted[0])

# Create confusion matrix
matrix = confusion_matrix(target_test, target_predicted)
# Create pandas dataframe
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
# Create a classification report
print(classification_report(target_test,target_predicted,target_names=class_names))

# Create heatmap
sns.heatmap(dataframe,annot=True, cbar=None, cmap="Blues")
plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()

# Create confusion matrix
matrix = confusion_matrix(target_test, target_predicted)
# Create pandas dataframe
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
# Create a classification report
print(classification_report(target_test,target_predicted,target_names=class_names))

# Create heatmap
sns.heatmap(dataframe,annot=True, cbar=None, cmap="Blues")
plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()

""" Bu kısım kategorik veriler için değil cont. veriler için daha uygundur. Fakat bu proje kapsamında yalnızca kategorik veriler için incelemeye vaktimiz oldu.
#Aşağıdaki fonksiyonların her birinin kendisine ait birer kullanım alanı bulunmaktadır.
#Bernoulli

#Multinominal
bnb = MultinomialNB()
bnb.fit(features_train, target_train)
print('Accuracy of GNB classifier on training set: {:.2f}'.format(bnb.score(features_train, target_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'.format(bnb.score(features_test, target_test)))

#CategoricalNB
bnb = CategoricalNB()
bnb.fit(features_train, target_train)
print('Accuracy of GNB classifier on training set: {:.2f}'.format(bnb.score(features_train, target_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'.format(bnb.score(features_test, target_test)))

"""
