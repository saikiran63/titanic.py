
# predict if a passenger survived the sinking of the Titanic or not.


```python
#importing all required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
#load the train data
train=pd.read_csv("C:\\Users\\Saikiran\\Desktop\\project data sets\\titantic\\train.csv")
```


```python
train.shape
```




    (891, 12)




```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>886</th>
      <td>887</td>
      <td>0</td>
      <td>2</td>
      <td>Montvila, Rev. Juozas</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>211536</td>
      <td>13.00</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>887</th>
      <td>888</td>
      <td>1</td>
      <td>1</td>
      <td>Graham, Miss. Margaret Edith</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>112053</td>
      <td>30.00</td>
      <td>B42</td>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>889</td>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Miss. Catherine Helen "Carrie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.45</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>889</th>
      <td>890</td>
      <td>1</td>
      <td>1</td>
      <td>Behr, Mr. Karl Howell</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>111369</td>
      <td>30.00</td>
      <td>C148</td>
      <td>C</td>
    </tr>
    <tr>
      <th>890</th>
      <td>891</td>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.75</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Finding the null values in the dataset
```


```python
train.isnull().sum()
```




    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64




```python
#removing null values and replacing with median in the age column
```


```python
train['Age'].fillna(train['Age'].median(), inplace=True)
```


```python
train.isnull().sum()
```




    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age              0
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64




```python
train=train.drop('Name',1)
train=train.drop('Ticket',1)
train=train.drop('PassengerId',1)
train=train.drop('Cabin',1)
```


```python
train.isnull().sum()
```




    Survived    0
    Pclass      0
    Sex         0
    Age         0
    SibSp       0
    Parch       0
    Fare        0
    Embarked    2
    dtype: int64




```python
train["Embarked"].fillna('S',inplace=True)
```


```python
# Import label encoder 
from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'species'. 
train['Sex']= label_encoder.fit_transform(train['Sex']) 
```


```python
train['Embarked']= label_encoder.fit_transform(train['Embarked'])
```


```python
train.isnull().sum()
```




    Survived    0
    Pclass      0
    Sex         0
    Age         0
    SibSp       0
    Parch       0
    Fare        0
    Embarked    0
    dtype: int64




```python
train.shape
```




    (891, 8)




```python
# male=1, female=0 ana in Embarked 1=Q,0=C,2=S
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
#EXPLORATORY DATA ANALYSIS
#checking wheater the data set is balanced or not
```


```python
train['Survived'].value_counts()
```




    0    549
    1    342
    Name: Survived, dtype: int64




```python
sns.countplot(x='Survived',data=train)
plt.show()
```


![png](output_21_0.png)



```python
# Countplot 
sns.catplot(x ="Sex", hue ="Survived",  kind ="count", data = train) 
plt.show()
```


![png](output_22_0.png)



```python
sns.catplot(x='Sex', col='Survived', kind='count', data=train)
plt.show()
```


![png](output_23_0.png)



```python
sns.set_style("whitegrid");
sns.FacetGrid(train, hue="Survived", size=4) \
   .map(plt.scatter, "Sex", "Age") \
   .add_legend();
plt.show();
```

    C:\Users\Saikiran\Anaconda3\lib\site-packages\seaborn\axisgrid.py:230: UserWarning: The `size` paramter has been renamed to `height`; please update your code.
      warnings.warn(msg, UserWarning)
    


![png](output_24_1.png)



```python
sns.violinplot(x ="Sex", y ="Age", hue ="Survived",  data = train, split = True)
plt.show()
```


![png](output_25_0.png)



```python
#Feature engineering
```


```python
#Dropping the outlier rows with Percentiles
#upper_lim = train['Fare'].quantile(.97)
#lower_lim = train['Fare'].quantile(.03)

#train = train[(train['Fare'] < upper_lim) & (train['Fare'] > lower_lim)]
```


```python
sns.catplot(x ='Embarked', hue ='Survived',  
kind ='count', col ='Pclass', data = train)
```




    <seaborn.axisgrid.FacetGrid at 0x29b41758c88>




![png](output_28_1.png)



```python
train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>0.647587</td>
      <td>29.361582</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
      <td>1.536476</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>0.477990</td>
      <td>13.019697</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
      <td>0.791503</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>22.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>35.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
      <td>2.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python
from sklearn.preprocessing import StandardScaler
train[['Pclass','Sex','Age','SibSp','Parch', 'Fare','Embarked']] = StandardScaler().fit_transform(train[['Pclass','Sex','Age','SibSp','Parch', 'Fare','Embarked']])
```


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.827377</td>
      <td>0.737695</td>
      <td>-0.565736</td>
      <td>0.432793</td>
      <td>-0.473674</td>
      <td>-0.502445</td>
      <td>0.585954</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-1.566107</td>
      <td>-1.355574</td>
      <td>0.663861</td>
      <td>0.432793</td>
      <td>-0.473674</td>
      <td>0.786845</td>
      <td>-1.942303</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0.827377</td>
      <td>-1.355574</td>
      <td>-0.258337</td>
      <td>-0.474545</td>
      <td>-0.473674</td>
      <td>-0.488854</td>
      <td>0.585954</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>-1.566107</td>
      <td>-1.355574</td>
      <td>0.433312</td>
      <td>0.432793</td>
      <td>-0.473674</td>
      <td>0.420730</td>
      <td>0.585954</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0.827377</td>
      <td>0.737695</td>
      <td>0.433312</td>
      <td>-0.474545</td>
      <td>-0.473674</td>
      <td>-0.486337</td>
      <td>0.585954</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.shape
```




    (891, 8)




```python
#Loading test data
```


```python
test=pd.read_csv("C:\\Users\\Saikiran\\Desktop\\project data sets\\titantic\\test.csv")
```


```python
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.shape
```




    (418, 11)




```python
test.isnull().sum()
```




    PassengerId      0
    Pclass           0
    Name             0
    Sex              0
    Age             86
    SibSp            0
    Parch            0
    Ticket           0
    Fare             1
    Cabin          327
    Embarked         0
    dtype: int64




```python
test['Age'].fillna(train['Age'].median(), inplace=True)
```


```python
test=test.drop('Name',1)
test=test.drop('Ticket',1)
test=test.drop('PassengerId',1)
test=test.drop('Cabin',1)
```


```python
test.isnull().sum()
```




    Pclass      0
    Sex         0
    Age         0
    SibSp       0
    Parch       0
    Fare        1
    Embarked    0
    dtype: int64




```python
test['Fare'].fillna(test['Fare'].median(), inplace=True)
```


```python
test['Fare'].describe()
```




    count    418.000000
    mean      35.576535
    std       55.850103
    min        0.000000
    25%        7.895800
    50%       14.454200
    75%       31.471875
    max      512.329200
    Name: Fare, dtype: float64




```python
test.shape
```




    (418, 7)




```python
test.isnull().sum()
```




    Pclass      0
    Sex         0
    Age         0
    SibSp       0
    Parch       0
    Fare        0
    Embarked    0
    dtype: int64




```python
# Import label encoder 
from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'species'. 
test['Sex']= label_encoder.fit_transform(test['Sex']) 
```


```python
test['Embarked']= label_encoder.fit_transform(test['Embarked'])
```


```python
test.isnull().sum()
```




    Pclass      0
    Sex         0
    Age         0
    SibSp       0
    Parch       0
    Fare        0
    Embarked    0
    dtype: int64




```python
test.shape
```




    (418, 7)




```python
from sklearn.preprocessing import StandardScaler
test[['Pclass','Sex','Age','SibSp','Parch', 'Fare','Embarked']] = StandardScaler().fit_transform(test[['Pclass','Sex','Age','SibSp','Parch', 'Fare','Embarked']])
```


```python
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.873482</td>
      <td>0.755929</td>
      <td>0.595028</td>
      <td>-0.499470</td>
      <td>-0.400248</td>
      <td>-0.497413</td>
      <td>-0.470915</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.873482</td>
      <td>-1.322876</td>
      <td>1.304932</td>
      <td>0.616992</td>
      <td>-0.400248</td>
      <td>-0.512278</td>
      <td>0.700767</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.315819</td>
      <td>0.755929</td>
      <td>2.156817</td>
      <td>-0.499470</td>
      <td>-0.400248</td>
      <td>-0.464100</td>
      <td>-0.470915</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.873482</td>
      <td>0.755929</td>
      <td>0.169086</td>
      <td>-0.499470</td>
      <td>-0.400248</td>
      <td>-0.482475</td>
      <td>0.700767</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.873482</td>
      <td>-1.322876</td>
      <td>-0.114876</td>
      <td>0.616992</td>
      <td>0.619896</td>
      <td>-0.417492</td>
      <td>0.700767</td>
    </tr>
  </tbody>
</table>
</div>




```python
# modeling
```


```python
xtrain=train
```


```python
xtrain.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.827377</td>
      <td>0.737695</td>
      <td>-0.565736</td>
      <td>0.432793</td>
      <td>-0.473674</td>
      <td>-0.502445</td>
      <td>0.585954</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-1.566107</td>
      <td>-1.355574</td>
      <td>0.663861</td>
      <td>0.432793</td>
      <td>-0.473674</td>
      <td>0.786845</td>
      <td>-1.942303</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0.827377</td>
      <td>-1.355574</td>
      <td>-0.258337</td>
      <td>-0.474545</td>
      <td>-0.473674</td>
      <td>-0.488854</td>
      <td>0.585954</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>-1.566107</td>
      <td>-1.355574</td>
      <td>0.433312</td>
      <td>0.432793</td>
      <td>-0.473674</td>
      <td>0.420730</td>
      <td>0.585954</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0.827377</td>
      <td>0.737695</td>
      <td>0.433312</td>
      <td>-0.474545</td>
      <td>-0.473674</td>
      <td>-0.486337</td>
      <td>0.585954</td>
    </tr>
  </tbody>
</table>
</div>




```python
x_train=train.drop('Survived',axis=1)
y_train=train['Survived']
```


```python
#Applying model to train the data sets
```


```python
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors=7) 
knn.fit(x_train, y_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=7, p=2,
                         weights='uniform')




```python
y_pred = knn.predict(test)
y_pred
```




    array([0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0,
           1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1,
           1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1,
           1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1,
           1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
           0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
           1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
           0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0,
           1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1,
           0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1,
           0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
           0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0,
           0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0,
           1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0,
           0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0,
           1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
           0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
          dtype=int64)




```python
#load the ytest data
y=pd.read_csv("C:\\Users\\Saikiran\\Desktop\\project data sets\\titantic\\gender_submission.csv")
y_test=y.drop('PassengerId',axis=1)
```


```python
y_test.shape
```




    (418, 1)




```python
# comparing actual response values (y_test) with predicted response values (y_pred) 
from sklearn import metrics 
print("kNN model accuracy:", metrics.accuracy_score(y_test, y_pred))
```

    kNN model accuracy: 0.8755980861244019
    


```python
# train the model
knn.fit(x_train, y_train)
# get the predict value from X_test
y_pred = knn.predict(test)
# print the score
print('accuracy: ', knn.score(test, y_test))
```

    accuracy:  0.8755980861244019
    


```python
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
print('Validation Results')
print('Accuracy Score: ', accuracy_score(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
confusion_matrix(y_test, y_pred)
```

    Validation Results
    Accuracy Score:  0.8755980861244019
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.90      0.91      0.90       266
               1       0.84      0.82      0.83       152
    
        accuracy                           0.88       418
       macro avg       0.87      0.86      0.86       418
    weighted avg       0.88      0.88      0.88       418
    
    Confusion Matrix:
    




    array([[242,  24],
           [ 28, 124]], dtype=int64)




```python
#CONFUSION MATRIX
```


```python
!pip install -q scikit-plot
import scikitplot as skplt

skplt.metrics.plot_confusion_matrix(y_test, y_pred,figsize=(9,9))
plt.show()
```


![png](output_65_0.png)



```python
#Ensemble models
```


```python

```


```python
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
```


```python
# Bagged Decision Trees for Classification
```


```python
kfold = model_selection.KFold(n_splits=10, random_state=7)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=7)
results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold)
print(results.mean())
```

    0.817116104868914
    


```python
# AdaBoost Classification
```


```python
from sklearn.ensemble import AdaBoostClassifier
seed = 7
num_trees = 100
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold)
print(results.mean())
```

    0.8103870162297128
    


```python
# Voting Ensemble for Classification
```


```python
import warnings
warnings.filterwarnings('ignore')
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

kfold = model_selection.KFold(n_splits=10, random_state=1)
# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble,x_train, y_train, cv=kfold)
print(results.mean())
```

    0.8271660424469414
    


```python

```
