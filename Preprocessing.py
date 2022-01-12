import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer, StandardScaler
import csv


plt.style.use('seaborn')

##İleride eğitim seti için ayırmakta kullanabliriz
#train_data,test_data=train_test_split(playstore,test_size=0.15,random_state=42)

# Helper function: Draw Hist Plot
def num_plots(df, col, title, xlabel):
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8,5))
    ax[0].set_title(title,fontsize=18)
    
    mean_text = 'Mean of ', col, ' is: ', df[col].mean()
    median_text = 'Median of ', col, ' is: ', df[col].median()
    mode_text = 'Mode of ', col, ' is: ', df[col].mode()[0]
    '''
    plt.text(0, 1, mean_text, fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
    plt.text(0, 3, median_text, fontsize=12, bbox=dict(facecolor='green', alpha=0.5))
    plt.text(0, 5, mode_text, fontsize=12, bbox=dict(facecolor='blue', alpha=0.5))
    '''
    print(mean_text, '\n', median_text, '\n', mode_text, '\n')

    sns.boxplot(x=col, data=df, ax=ax[0])
    ax[0].set(yticks=[])
    sns.histplot(x=col, data=df, ax=ax[1])
    ax[1].set_xlabel(xlabel, fontsize=16)

    plt.tight_layout()
    plt.show()

# Helper function: Draw Box Plot
def draw_boxplot(df, feature, vers):
  sns.boxplot(x=feature, data=df)
  title = feature + ' (' + vers + ' dropping outliers)'
  plt.title(title);
  plt.show()

# Missing Values
def missing_values(df):
    #check all missing values in dataset  
    print("Number of rows having null values in the dataset:")
    missing_info = (len(df[df.isnull().any(axis=1)]) / len(df) )*100
    print(len(df[df.isnull().any(axis=1)]),' which is ' ,round(missing_info,2) , '%')
    
    #check missing values in columns
    cols = df.columns[df.isnull().any()].to_list()
    print("Columns having null values are :",cols)

    for c in cols:
      missing_info=((df[c].isnull().sum()) / len(df[c]) )*100
      print(c,type(c),": ",df[c].isnull().sum(), "," ,round(missing_info,2) , '%')

    #drop if the missing value count is less than %1
    df.dropna(subset=['Installs'],inplace=True)
    df.dropna(subset=['Size'],inplace=True)
    df.dropna(subset=['Minimum Android'],inplace=True)
    
    #Handling Rating and Rating Count missing values
    df['Rating']  = df['Rating'].astype(float)
    avg = round(df['Rating'].mean(),1)
    df['Rating'].fillna(avg,inplace=True)

    df['Rating Count']  = df['Rating Count'].astype(float)
    avg = round(df['Rating Count'].mean(),1)
    df['Rating Count'].fillna(avg,inplace=True)

    
# Detect Outliers
def detect_and_drop_outliers(feature, df):
  q1 = df[feature].quantile(0.25)
  q3 = df[feature].quantile(0.75)  
  iqr = q3 - q1
  lower_bound = q1 - (iqr * 1.5)
  upper_bound = q3 + (iqr * 1.5)
  return df[~( (df[feature] < lower_bound) | (df[feature] > upper_bound) )]

def main():
  # read dataset
  df = pd.read_csv('./dataset/Google-Playstore.csv')

  # drop duplicate values
  df.drop_duplicates(subset='App Name', inplace=True, ignore_index=True)

  # drop unnecessary columns
  df.drop('App Id',inplace=True,axis=1)
  df.drop('App Name',inplace=True,axis=1)
  df.drop('Currency',inplace=True,axis=1)
  df.drop('Developer Email',inplace=True,axis=1)
  df.drop('Developer Id',inplace=True,axis=1)
  df.drop('Developer Website',inplace=True,axis=1)
  df.drop('Price',inplace=True,axis=1)
  df.drop('Privacy Policy',inplace=True,axis=1)
  df.drop('Scraped Time',inplace=True,axis=1)
  df.drop('Maximum Installs',inplace=True,axis=1)
  df.drop('Minimum Installs',inplace=True,axis=1)
  
  df.head()
  #print("Dataset information", df.info())  

  # Handle Missing Values
  missing_values(df)
  df_clean = df.copy()

  ### Continuous Features ###

  # Size Column 
  ''' size values that corresponds to 'Varies with device' with 'NaN' '''
  df_clean['Size'] = df_clean['Size'].replace('Varies with device', 'NaN', regex=True)
  df_clean['Size'] = df_clean['Size'].replace('nan', 'NaN', regex=True)
  
  df_clean['Minimum Android'] = df_clean['Minimum Android'].str.replace('Var','4.1',regex=True)

  ''' decimal point ',' to '.' '''
  df_clean['Size'] = df_clean['Size'].str.replace(',','.')

  ''' convert unit '''
  size = []
  
  for i in df_clean['Size']:
    if i == 'NaN':
      size.append('NaN')
    elif i[-1] == 'k' or i[-1] == 'K':
      size.append(float(i[:-1])/1000)
    elif i[-1] == 'G' or i[-1] == 'g':
      size.append(float(i[:-1])*1000)
    else:
      size.append(float(i[:-1]))
  
  ''' fix units of Size '''
  df_clean['Size'] = size
  df_clean['Size'] = df_clean['Size'].astype(float)

  ''' Handling Size Missing Values '''
  df_clean['Size']=df_clean['Size'].fillna(df_clean["Size"].mode()[0])

  # Installs Column
  df_clean.Installs = df_clean.Installs.str.replace(',','',regex=True)
  df_clean.Installs = df_clean.Installs.str.replace('+','',regex=True)
  df_clean.Installs = df_clean.Installs.str.replace('Free','0',regex=True)
  df_clean['Installs'] = pd.to_numeric(df_clean['Installs'])
  
  print( df_clean['Installs'].value_counts())
  #  Normalization
  ''' Continuous Features are: 'Size', Install-> başlarda Install değeri ile de denendi fakat sonrasında target attr. olduğu için değiştirildi  '''
  #cont_features = ['Size']

  '''
  for feature in cont_features:
   
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1))
    will_normalize_feature = np.array(df_clean[feature])
    scaled_feature = minmax_scale.fit_transform(will_normalize_feature)

    scaled_feature = np.array(scaled_feature)
    df_clean[feature] = StandardScaler().fit_transform(scaled_feature)
  '''

  ''' least squares normalizer form '''
  #normalizer = Normalizer(norm="l2")
  #df_clean[cont_features] = normalizer.transform(df_clean[cont_features])

  """
  # Find and Clean Outliers
  '''  boxplot before drop outliers '''
  for feature in cont_features:
    draw_boxplot(df_clean, feature, 'before')

  ''' drop outliers '''
  for feature in cont_features:
    df_clean = detect_and_drop_outliers(feature, df_clean)

  '''  boxplot after drop outliers '''
  for feature in cont_features:
    draw_boxplot(df_clean, feature, 'after')
  """

  ### Categorical Features ###

  # Content Rating
  '''  print(df_clean['Content Rating'].value_counts()) '''
  df_clean['Content Rating'] = df_clean['Content Rating'].replace('Unrated',"Everyone")

  '''  Cleaning other values just to include Everyone, Teens and Adult '''
  df_clean['Content Rating'] = df_clean['Content Rating'].replace('Mature 17+',"Adults")
  df_clean['Content Rating'] = df_clean['Content Rating'].replace('Adults only 18+',"Adults")
  df_clean['Content Rating'] = df_clean['Content Rating'].replace('Everyone 10+',"Everyone")
  
  conditions = [
    (df_clean['Rating'] == 0) & (df_clean['Rating Count'] == 0),
    (df_clean['Rating'] >= 0) & (df_clean['Rating'] <= 1) & (df_clean['Rating Count'] != 0),
    (df_clean['Rating'] > 1) & (df_clean['Rating'] <= 2) & (df_clean['Rating Count'] != 0),
    (df_clean['Rating'] > 2) & (df_clean['Rating'] <= 3) & (df_clean['Rating Count'] != 0),
    (df_clean['Rating'] > 3) & (df_clean['Rating'] <= 4) & (df_clean['Rating Count'] != 0),
    (df_clean['Rating'] > 4) & (df_clean['Rating'] <= 5) & (df_clean['Rating Count'] != 0)
    ]

  # create a list of the values we want to assign for each condition
  values = ['Unranked', 'Very Bad', 'Bad', 'Not Bad','Good','Very Good']

  # create a new column and use np.select to assign values to it using our lists as arguments
  df_clean['AppRating'] = np.select(conditions, values)
  
  df_clean = df_clean.drop(['Rating','Rating Count'], axis = 1)
  
  print(len(df_clean['Category'].value_counts()))
  # Release Date and Update Date  
  ''' Burada COVID19 1 aralıkta başlamış kabul edilmiş ama ben onu 2020 de başlamış olarak kabul ediyorum. '''
  df_clean['Released'] = pd.to_datetime(df_clean['Released'], format='%b %d, %Y',infer_datetime_format=True, errors='coerce') 
  df_clean['Last Updated'] = pd.to_datetime(df_clean['Last Updated'], format='%b %d, %Y',infer_datetime_format=True, errors='coerce') 
  u = df_clean.select_dtypes(include=['datetime'])
  md=df_clean['Released'].median()
  df_clean[u.columns]=u.fillna(md)

  covid= []
  
  for i in pd.DatetimeIndex(df_clean['Released']).year:
      if i >= 2020:
        covid.append(True)
      else:
        covid.append(False)

  lastupdate = []
  
  for i in pd.DatetimeIndex(df_clean['Last Updated']).year:
      if i > 2020:
        lastupdate.append(True)
      else:
        lastupdate.append(False)
 
  #append Up to Date
  df_clean['Up to Date'] = lastupdate
  
  #append covid feature
  df_clean['Covid'] = covid
  
  # Minimum Android Version
  min_andr_ver = []
  ver_mode = df_clean['Minimum Android'].mode()[0]
  for i in df_clean['Minimum Android']:
    ver = i.split()
    if ver =='Varies with Device':
      min_andr_ver.append(ver_mode)
    else:
      min_andr_ver.append(ver[0][:3])
          
  df_clean['Minimum Android'] = min_andr_ver
  df_clean['Size'] = df_clean['Size'].astype(float)

  #Free
  df_clean['Type'] = np.where(df_clean['Free'] == True,'Free','Paid')
  df_clean.drop(['Free'],inplace=True,axis=1)
  
  
  
  #print("Dataset information",df_clean.info())  
  print(df_clean['Minimum Android'].value_counts())
  # open the file in the write mode
  df_clean.drop('Last Updated',inplace=True,axis=1)
  
  months= []
  
  for i in pd.DatetimeIndex(df_clean['Released']).month:
      if i == 1:
        months.append('January')
      elif i == 2:
        months.append('February')
      elif i == 3:
        months.append('March')
      elif i == 4:
        months.append('April')
      elif i == 5:
        months.append('May')
      elif i == 6:
        months.append('June')  
      elif i == 7:
        months.append('July')
      elif i == 8:
        months.append('August')
      elif i == 9:
        months.append('September')
      elif i == 10:
        months.append('October')
      elif i == 11:
        months.append('November')
      elif i == 12:
        months.append('December')        

  #append Up to Date
  df_clean['month'] = months
  
  df_clean.drop('Released',inplace=True,axis=1)
  
  df_clean.to_csv('./dataset/clean_googleplaystore_dataset.csv')
 

  ### Visualization ###
  '''
  # Install
  sns.countplot(x='Installs', data=df_clean)
  plt.title('Installs')
  plt.xticks(rotation=60)
  plt.show()
  # Rating Column
  '''
  '''
  num_plots(df_clean,'Rating','App rating distribution','Rating') '''

  # Categories
  '''
  sns.countplot(x='Category', data=df_clean, order = df_clean['Category'].value_counts().index)
  plt.xticks(rotation=90);
  plt.xlabel('')
  plt.title('App category counts');
  plt.show() '''

  # App Types
  '''
  sns.countplot(x='Free', data=df_clean)
  plt.title('Free or not')
  plt.xlabel('App type')
  plt.show()
  '''
 
  # Last Updated
  '''
  plt.figure(figsize=(10,4))
  sns.histplot(x='Last Updated', data=df_clean)
  plt.show()
  '''

  # Content Rating
  '''
  sns.countplot(x='Content Rating', data=df_clean)
  plt.title('Content Rating')
  plt.xticks(rotation=60)
  plt.show()
  '''

  # Minimum Android
  '''
  sns.countplot(x='Minimum Android', data=df_clean)
  plt.title('Minimum Android')
  plt.xticks(rotation=60)
  plt.show()
  '''

  # Released
  '''
  sns.countplot(x='Released', data=df_clean)
  plt.title('Released')
  plt.xticks(rotation=60)
  plt.show()
  '''

  # Ad Supported
  '''
  sns.countplot(x='Ad Supported', data=df_clean)
  plt.title('Ad Supported')
  plt.xticks(rotation=60)
  plt.show()
  '''

  # In App Purchases
  '''
  sns.countplot(x='In App Purchases', data=df_clean)
  plt.title('In App Purchases')
  plt.xticks(rotation=60)
  plt.show()
  '''

  # Editors Choice
  '''
  sns.countplot(x='Editors Choice', data=df_clean)
  plt.title('Editors Choice')
  plt.xticks(rotation=60)
  plt.show()
  '''


  # Released
  ''' Visualization
  sns.countplot(x='Released', data=df_clean)
  plt.title('Released')
  plt.xticks(rotation=60)
  plt.show()
 
  released_date_install=pd.concat([df_clean['Installs'],df_clean['Released']],axis=1)
  plt.figure(figsize=(15,12))
  released_date_plot=released_date_install.set_index('Released').resample('3M').mean()
  released_date_plot.plot()
  plt.title('Released date Vs Installs',fontdict={'size':20,'weight':'bold'})
  plt.plot()
  '''

  #Covid-19 
  """
  plt.pie(df_clean['Covid'].value_counts(),radius=3,autopct='%0.2f%%',explode=[0.2,0.5],colors=['#ffa500','#0000a0'],labels=['Before Covid','Covid'],startangle=90,textprops={'fontsize': 30})
  plt.title('Covid-19 Impact on Applications',fontdict={'size':20,'weight':'bold'})
  plt.plot()
  """
  #Up to Date
  """
  plt.pie(df_clean['Up to Date'].value_counts(),radius=3,autopct='%0.2f%%',explode=[0.2,0.5],colors=['#ffa500','#0000a0'],labels=['No','Yes'],startangle=90,textprops={'fontsize': 30})
  plt.title('Is the app up to date?',fontdict={'size':20,'weight':'bold'})
  plt.plot()
  """

    #Type vs Install
  '''
  plt.figure(figsize=(18,18))
  ax = sns.countplot(df_clean['Installs'],hue=df_clean['Type']);
  plt.title("Number of Installs in different Types ")

  plt.xticks(fontsize=10,fontweight='bold',rotation=45,ha='right');
  plt.show()
  '''

  # Last Updated
  '''
  plt.figure(figsize=(10,4))
  sns.histplot(x='Last Updated', data=df_clean)
  plt.show()
 
  lastupdate_install=pd.concat([df_clean['Installs'],df_clean['Last Updated']],axis=1)
  plt.figure(figsize=(15,12))
  released_date_plot=lastupdate_install.set_index('Last Updated').resample('3M').mean()
  released_date_plot.plot()
  plt.title('Last Update Vs Installs',fontdict={'size':20,'weight':'bold'})
  plt.plot()
  '''

  # Content Rating
  '''
  age_install = df_clean.groupby('Content Rating')['Minimum Installs'].mean()

  plt.axes().set_facecolor("white")
  plt.rcParams.update({'font.size': 12, 'figure.figsize': (5, 4)})
  plt.ylabel('Category')
  plt.xlabel('Installs per 10 million')
  age_install.sort_index().plot(kind="barh", title='Average Number of Installs per Content Rating');
  plt.gca().invert_yaxis()
  plt.savefig("Age rating", transparent=False, bbox_inches="tight")
  '''

  #Categories vs Install
  #draw a boxplot map to observe app's ratings among different categories
  """
  category_rating = df.groupby(['Category'])['Installs'].count()

  plt.figure(figsize=(15,10))
  sns.barplot(category_rating.index, category_rating.values)
  plt.title('Number of Installs Per Category')
  plt.xlabel('Category')
  plt.ylabel('Installs')
  plt.xticks(fontsize=10,fontweight='bold',rotation=45,ha='right');
  """
  """
  f, ax = plt.subplots(2,2,figsize=(10,15))

  ax[0,0].hist(df_clean.Rating, range=(3,5))
  ax[0,0].set_title('Ratings Histogram')
  ax[0,0].set_xlabel('Ratings')

  d = df_clean.groupby('Category')['Rating'].mean().reset_index()
  ax[0,1].scatter(d.Category, d.Rating)
  ax[0,1].set_xticklabels(d.Category.unique(),rotation=90)
  ax[0,1].set_title('Mean Rating per Category')

  ax[1,1].hist(df_clean.Size, range=(0,100),bins=10, label='Size')
  ax[1,1].set_title('Size Histogram')
  ax[1,1].set_xlabel('Size')

  d = df_clean.groupby('Size')['Installs'].mean().reset_index()
  ax[1,0].scatter(d.Size, d.Installs)
  ax[1,0].set_xticklabels(d.Size.unique(),rotation=90)
  ax[1,0].set_title('Mean Install per Size')
  f.tight_layout()
  """
  # Correlation Matrix
  sns.heatmap(df_clean.corr(), annot=True, cmap='Blues')
  plt.title('Correlation Matrix')
  plt.show()


if __name__ == "__main__":
    main()