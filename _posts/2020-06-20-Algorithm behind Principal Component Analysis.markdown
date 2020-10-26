---
title: Algorithm behind Principal Component Analysis
date: 2020-06-20 14:41:57 Z
categories:
- machine-learning
tags:
- machine learning
- regression
- scikit-learn
- PCA
layout: post
comments: true
description: Principal Component Analysis Visualizations using Python
author: Rasik Kane
---


### Motive behind Data pre-processing
Motive behind Data Processing or Mining is obtaining data and process it to create models for analysis. Insights obtained from these models support decision-making process. Before modelling, incomplete data as well as data which does not contribute towards analysis are pre-processed using: 
* Normalization
* Dimensionality reduction

### This notebook describes :
* **How PCA is implemnted manually**
* **How PCA boosts classsfication performance using only 50% of dimnesions of Input features**

## Principal Component Analysis

**Dimensionality Reduction** is data transformation technique that is used to reduce multidimensional data sets to a lower number of dimensions for further analysis. Its goal is to extract the important information from the database.
<br>


**Techniques for Dimensionality Reduction**
* **Elimination** : Drop variables with lesser correlation with target variable
    * This technique achieves results, but we dropped dimensions. Their effect on target variable [however minimal] is completely unaccounted for
* **Extraction** : Analyze varaibles and EXTRACT NEW varaibles [dimensions] from them. Insights from all variables are preserved and variance in target variable can be completely described these new dimensions.
    * This is methodlogy behind PCA : To  **Decompose** data sets as a function of the variance in the data
    * **PCA represents extracted information as a set of new orthogonal variables called principal components, and to display the pattern of similarity of the observations and of the variables as points in maps.**

### Import necessary libraries


```python
import warnings
import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression

%matplotlib inline
warnings.filterwarnings('ignore')
```

### Read red wine data


```python
# references: 
# https://stackoverflow.com/questions/32400867/pandas-read-csv-from-url
# https://svaderia.github.io/articles/downloading-and-unzipping-a-zipfile/
import io
import requests
url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
s=requests.get(url).content
df_red_wine=pd.read_csv(io.StringIO(s.decode('utf-8')), sep=";")
```


```python
print("shape of red wine data: ", df_red_wine.shape, "\n")
df_red_wine.head()
```

    shape of red wine data:  (1599, 12) 
    
    




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
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2</td>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.9980</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_red_wine['quality'].describe()
```




    count    1599.000000
    mean        5.636023
    std         0.807569
    min         3.000000
    25%         5.000000
    50%         6.000000
    75%         6.000000
    max         8.000000
    Name: quality, dtype: float64



### Convert to a classification problem


```python
df_red_wine['quality'] = [1 if quality >= 6 else 0 for quality in df_red_wine['quality']]
```

### Distribution of data
**Right skewed normal distribution** is genral tendancy of data.  


```python
sns.color_palette("Set2")
for col in df_red_wine.columns:
    g = sns.FacetGrid(df_red_wine, col="quality")
    g.map(sns.histplot, col)
```


    
<img src="/images/p1/output_13_0.png">
    



    
<img src="/images/p1/output_13_1.png">
    



    
<img src="/images/p1/output_13_2.png">
    



    
<img src="/images/p1/output_13_3.png">
    



    
<img src="/images/p1/output_13_4.png">
    



    
<img src="/images/p1/output_13_5.png">
    



    
<img src="/images/p1/output_13_6.png">
    



    
<img src="/images/p1/output_13_7.png">
    



    
<img src="/images/p1/output_13_8.png">
    



    
<img src="/images/p1/output_13_9.png">
    



    
<img src="/images/p1/output_13_10.png">
    



    
<img src="/images/p1/output_13_11.png">
    


### Normalization of input features
* Target feature 'quality' is ommited to form set of input features X
* PCA "extracts" i.e. converts input features into Principal compoents which Maximize Variance
    * **Standardisation i.e. Z-score normalisation is performed on data. It scales data to N(0,1)**
    * Majority of data is normally distributed, so standardization is good choice


```python
X = df_red_wine.drop(['quality'], axis=1)
```


```python
X_std = StandardScaler().fit_transform(X.values.astype('float64'))
X = pd.DataFrame(X_std, index=X.index, columns=X.columns)
X.head()
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
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.528360</td>
      <td>0.961877</td>
      <td>-1.391472</td>
      <td>-0.453218</td>
      <td>-0.243707</td>
      <td>-0.466193</td>
      <td>-0.379133</td>
      <td>0.558274</td>
      <td>1.288643</td>
      <td>-0.579207</td>
      <td>-0.960246</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.298547</td>
      <td>1.967442</td>
      <td>-1.391472</td>
      <td>0.043416</td>
      <td>0.223875</td>
      <td>0.872638</td>
      <td>0.624363</td>
      <td>0.028261</td>
      <td>-0.719933</td>
      <td>0.128950</td>
      <td>-0.584777</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.298547</td>
      <td>1.297065</td>
      <td>-1.186070</td>
      <td>-0.169427</td>
      <td>0.096353</td>
      <td>-0.083669</td>
      <td>0.229047</td>
      <td>0.134264</td>
      <td>-0.331177</td>
      <td>-0.048089</td>
      <td>-0.584777</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.654856</td>
      <td>-1.384443</td>
      <td>1.484154</td>
      <td>-0.453218</td>
      <td>-0.264960</td>
      <td>0.107592</td>
      <td>0.411500</td>
      <td>0.664277</td>
      <td>-0.979104</td>
      <td>-0.461180</td>
      <td>-0.584777</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.528360</td>
      <td>0.961877</td>
      <td>-1.391472</td>
      <td>-0.453218</td>
      <td>-0.243707</td>
      <td>-0.466193</td>
      <td>-0.379133</td>
      <td>0.558274</td>
      <td>1.288643</td>
      <td>-0.579207</td>
      <td>-0.960246</td>
    </tr>
  </tbody>
</table>
</div>



### Statistics of normalized Input features 
* As observed, all features have mean ~ 0; standard deviation ~ 1
* min and max of features vary as per distance of a datapoint from mean varies. Effect of outliers can not be compensated for in Norm(0,1).


```python
X.describe()
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
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.599000e+03</td>
      <td>1.599000e+03</td>
      <td>1.599000e+03</td>
      <td>1.599000e+03</td>
      <td>1.599000e+03</td>
      <td>1.599000e+03</td>
      <td>1.599000e+03</td>
      <td>1.599000e+03</td>
      <td>1.599000e+03</td>
      <td>1.599000e+03</td>
      <td>1.599000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.554936e-16</td>
      <td>1.733031e-16</td>
      <td>-8.887339e-17</td>
      <td>-1.244227e-16</td>
      <td>3.910429e-16</td>
      <td>-6.221137e-17</td>
      <td>4.443669e-17</td>
      <td>2.364032e-14</td>
      <td>2.861723e-15</td>
      <td>6.754377e-16</td>
      <td>1.066481e-16</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.000313e+00</td>
      <td>1.000313e+00</td>
      <td>1.000313e+00</td>
      <td>1.000313e+00</td>
      <td>1.000313e+00</td>
      <td>1.000313e+00</td>
      <td>1.000313e+00</td>
      <td>1.000313e+00</td>
      <td>1.000313e+00</td>
      <td>1.000313e+00</td>
      <td>1.000313e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.137045e+00</td>
      <td>-2.278280e+00</td>
      <td>-1.391472e+00</td>
      <td>-1.162696e+00</td>
      <td>-1.603945e+00</td>
      <td>-1.422500e+00</td>
      <td>-1.230584e+00</td>
      <td>-3.538731e+00</td>
      <td>-3.700401e+00</td>
      <td>-1.936507e+00</td>
      <td>-1.898919e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-7.007187e-01</td>
      <td>-7.699311e-01</td>
      <td>-9.293181e-01</td>
      <td>-4.532184e-01</td>
      <td>-3.712290e-01</td>
      <td>-8.487156e-01</td>
      <td>-7.440403e-01</td>
      <td>-6.077557e-01</td>
      <td>-6.551405e-01</td>
      <td>-6.382196e-01</td>
      <td>-8.663789e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-2.410944e-01</td>
      <td>-4.368911e-02</td>
      <td>-5.636026e-02</td>
      <td>-2.403750e-01</td>
      <td>-1.799455e-01</td>
      <td>-1.793002e-01</td>
      <td>-2.574968e-01</td>
      <td>1.760083e-03</td>
      <td>-7.212705e-03</td>
      <td>-2.251281e-01</td>
      <td>-2.093081e-01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.057952e-01</td>
      <td>6.266881e-01</td>
      <td>7.652471e-01</td>
      <td>4.341614e-02</td>
      <td>5.384542e-02</td>
      <td>4.901152e-01</td>
      <td>4.723184e-01</td>
      <td>5.768249e-01</td>
      <td>5.759223e-01</td>
      <td>4.240158e-01</td>
      <td>6.354971e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.355149e+00</td>
      <td>5.877976e+00</td>
      <td>3.743574e+00</td>
      <td>9.195681e+00</td>
      <td>1.112703e+01</td>
      <td>5.367284e+00</td>
      <td>7.375154e+00</td>
      <td>3.680055e+00</td>
      <td>4.528282e+00</td>
      <td>7.918677e+00</td>
      <td>4.202453e+00</td>
    </tr>
  </tbody>
</table>
</div>



<br>
<br>

## PCA calculation : Steps
* Calculating the covariance matrix
* Calculating the eigenvalues and eigenvector
* Forming Principal Components
* Projection into the new feature space

### Covariance matrix
**Dimensions {X1, X2,..Xn} = {fixed acidity, volatile acidity, citric acid, residual sugar, chlorides....pH, sulphates, alcohol}**

Cov(XX) is given by
\begin{bmatrix}\mathrm {E} [(X_{1}-\operatorname {E} [X_{1}])(X_{1}-\operatorname {E} [X_{1}])]&\mathrm {E} [(X_{1}-\operatorname {E} [X_{1}])(X_{2}-\operatorname {E} [X_{2}])]&\cdots &\mathrm {E} [(X_{1}-\operatorname {E} [X_{1}])(X_{n}-\operatorname {E} [X_{n}])]\\\\\mathrm {E} [(X_{2}-\operatorname {E} [X_{2}])(X_{1}-\operatorname {E} [X_{1}])]&\mathrm {E} [(X_{2}-\operatorname {E} [X_{2}])(X_{2}-\operatorname {E} [X_{2}])]&\cdots &\mathrm {E} [(X_{2}-\operatorname {E} [X_{2}])(X_{n}-\operatorname {E} [X_{n}])]\\\\\vdots &\vdots &\ddots &\vdots \\\\\mathrm {E} [(X_{n}-\operatorname {E} [X_{n}])(X_{1}-\operatorname {E} [X_{1}])]&\mathrm {E} [(X_{n}-\operatorname {E} [X_{n}])(X_{2}-\operatorname {E} [X_{2}])]&\cdots &\mathrm {E} [(X_{n}-\operatorname {E} [X_{n}])(X_{n}-\operatorname {E} [X_{n}])]\end{bmatrix}


```python
cov_matrix = np.cov(X.T)
```


```python
cov_matrix[:3]
```




    array([[ 1.00062578, -0.25629118,  0.67212377,  0.11484855,  0.09376383,
            -0.15389043, -0.11325227,  0.66846534, -0.68340559,  0.18312019,
            -0.06170686],
           [-0.25629118,  1.00062578, -0.55284143,  0.00191908,  0.06133613,
            -0.0105104 ,  0.07651786,  0.02204002,  0.23508431, -0.26115001,
            -0.20241462],
           [ 0.67212377, -0.55284143,  1.00062578,  0.14366701,  0.20395046,
            -0.06101629,  0.03555526,  0.36517555, -0.54224326,  0.31296577,
             0.10997202]])



### Eigenvalues and Eigenvector


```python
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
```


```python
for val in eigenvalues:
    print(val)
```

    3.1010718226728273
    1.9271148896585149
    1.5515137913334218
    1.2139917499341308
    0.9598923792754817
    0.059595582455006985
    0.18144664164085156
    0.34485778773040704
    0.42322137844374963
    0.5841565453623766
    0.6600210359988645
    


```python
for val in eigenvectors.T:
    print(val)
```

    [ 0.48931422 -0.23858436  0.46363166  0.14610715  0.21224658 -0.03615752
      0.02357485  0.39535301 -0.43851962  0.24292133 -0.11323206]
    [-0.11050274  0.27493048 -0.15179136  0.27208024  0.14805156  0.51356681
      0.56948696  0.23357549  0.00671079 -0.03755392 -0.38618096]
    [-0.12330157 -0.44996253  0.23824707  0.10128338 -0.09261383  0.42879287
      0.3224145  -0.33887135  0.05769735  0.27978615  0.47167322]
    [-0.22961737  0.07895978 -0.07941826 -0.37279256  0.66619476 -0.04353782
     -0.03457712 -0.17449976 -0.00378775  0.55087236 -0.12218109]
    [-0.08261366  0.21873452 -0.05857268  0.73214429  0.2465009  -0.15915198
     -0.22246456  0.15707671  0.26752977  0.22596222  0.35068141]
    [-0.63969145 -0.0023886   0.0709103  -0.18402996 -0.05306532  0.05142086
     -0.0687016   0.5673319  -0.3407109  -0.06955538  0.31452591]
    [-0.24952314  0.36592473  0.62167708  0.09287208 -0.21767112  0.24848326
     -0.37075027 -0.23999012 -0.0109696   0.11232046 -0.3030145 ]
    [ 0.19402091 -0.1291103  -0.38144967  0.00752295  0.11133867  0.63540522
     -0.59211589  0.02071868 -0.16774589 -0.05836706  0.03760311]
    [-0.17759545 -0.07877531 -0.37751558  0.29984469 -0.35700936 -0.2047805
      0.01903597 -0.23922267 -0.56139075  0.37460432 -0.21762556]
    [-0.35022736 -0.5337351   0.10549701  0.29066341  0.37041337 -0.11659611
     -0.09366237 -0.17048116 -0.02513762 -0.44746911 -0.3276509 ]
    [ 0.10147858  0.41144893  0.06959338  0.04915555  0.30433857 -0.01400021
      0.13630755 -0.3911523  -0.52211645 -0.38126343  0.36164504]
    


```python
eigen_map = list(zip(eigenvalues, eigenvectors.T))
```


```python
eigen_map.sort(key=lambda x: x[0], reverse=True)
```


```python
sorted_eigenvalues = [pair[0] for pair in eigen_map]
sorted_eigenvectors = [pair[1] for pair in eigen_map]
```

### Formation of Principal Components


```python
sorted_eigenvalues
```




    [3.1010718226728273,
     1.9271148896585149,
     1.5515137913334218,
     1.2139917499341308,
     0.9598923792754817,
     0.6600210359988645,
     0.5841565453623766,
     0.42322137844374963,
     0.34485778773040704,
     0.18144664164085156,
     0.059595582455006985]




```python
pd.DataFrame(sorted_eigenvectors, columns=df_red_wine.drop(['quality'], axis=1).columns)
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
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.489314</td>
      <td>-0.238584</td>
      <td>0.463632</td>
      <td>0.146107</td>
      <td>0.212247</td>
      <td>-0.036158</td>
      <td>0.023575</td>
      <td>0.395353</td>
      <td>-0.438520</td>
      <td>0.242921</td>
      <td>-0.113232</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.110503</td>
      <td>0.274930</td>
      <td>-0.151791</td>
      <td>0.272080</td>
      <td>0.148052</td>
      <td>0.513567</td>
      <td>0.569487</td>
      <td>0.233575</td>
      <td>0.006711</td>
      <td>-0.037554</td>
      <td>-0.386181</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.123302</td>
      <td>-0.449963</td>
      <td>0.238247</td>
      <td>0.101283</td>
      <td>-0.092614</td>
      <td>0.428793</td>
      <td>0.322415</td>
      <td>-0.338871</td>
      <td>0.057697</td>
      <td>0.279786</td>
      <td>0.471673</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.229617</td>
      <td>0.078960</td>
      <td>-0.079418</td>
      <td>-0.372793</td>
      <td>0.666195</td>
      <td>-0.043538</td>
      <td>-0.034577</td>
      <td>-0.174500</td>
      <td>-0.003788</td>
      <td>0.550872</td>
      <td>-0.122181</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.082614</td>
      <td>0.218735</td>
      <td>-0.058573</td>
      <td>0.732144</td>
      <td>0.246501</td>
      <td>-0.159152</td>
      <td>-0.222465</td>
      <td>0.157077</td>
      <td>0.267530</td>
      <td>0.225962</td>
      <td>0.350681</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.101479</td>
      <td>0.411449</td>
      <td>0.069593</td>
      <td>0.049156</td>
      <td>0.304339</td>
      <td>-0.014000</td>
      <td>0.136308</td>
      <td>-0.391152</td>
      <td>-0.522116</td>
      <td>-0.381263</td>
      <td>0.361645</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.350227</td>
      <td>-0.533735</td>
      <td>0.105497</td>
      <td>0.290663</td>
      <td>0.370413</td>
      <td>-0.116596</td>
      <td>-0.093662</td>
      <td>-0.170481</td>
      <td>-0.025138</td>
      <td>-0.447469</td>
      <td>-0.327651</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-0.177595</td>
      <td>-0.078775</td>
      <td>-0.377516</td>
      <td>0.299845</td>
      <td>-0.357009</td>
      <td>-0.204781</td>
      <td>0.019036</td>
      <td>-0.239223</td>
      <td>-0.561391</td>
      <td>0.374604</td>
      <td>-0.217626</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.194021</td>
      <td>-0.129110</td>
      <td>-0.381450</td>
      <td>0.007523</td>
      <td>0.111339</td>
      <td>0.635405</td>
      <td>-0.592116</td>
      <td>0.020719</td>
      <td>-0.167746</td>
      <td>-0.058367</td>
      <td>0.037603</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.249523</td>
      <td>0.365925</td>
      <td>0.621677</td>
      <td>0.092872</td>
      <td>-0.217671</td>
      <td>0.248483</td>
      <td>-0.370750</td>
      <td>-0.239990</td>
      <td>-0.010970</td>
      <td>0.112320</td>
      <td>-0.303015</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-0.639691</td>
      <td>-0.002389</td>
      <td>0.070910</td>
      <td>-0.184030</td>
      <td>-0.053065</td>
      <td>0.051421</td>
      <td>-0.068702</td>
      <td>0.567332</td>
      <td>-0.340711</td>
      <td>-0.069555</td>
      <td>0.314526</td>
    </tr>
  </tbody>
</table>
</div>



### Explained Variance : choice of Principal compoents


```python
eigenvalue_sum = sum(eigenvalues)
var_exp = [(v / eigenvalue_sum)*100 for v in sorted_eigenvalues]
cum_var_exp = np.cumsum(var_exp)
```


```python
cum_var_exp
```




    array([ 28.17393128,  45.68220118,  59.77805108,  70.80743772,
            79.52827474,  85.52471351,  90.83190641,  94.67696732,
            97.81007747,  99.4585608 , 100.        ])




```python
dims = len(df_red_wine.drop(['quality'], axis=1).columns)
```


```python
plt.clf()
fig, ax = plt.subplots()

ax.plot(range(dims), cum_var_exp, '-o')

plt.xlabel('Number of Components')
plt.ylabel('Percent of Variance Explained')

plt.show()
```


    <Figure size 432x288 with 0 Axes>



    
<img src="/images/p1/output_37_1.png">
    


### It is noted that 6 eigenvectors describe more than 84% of varaince in target variable  


```python
ev1 = sorted_eigenvectors[0]
ev2 = sorted_eigenvectors[1]
ev3 = sorted_eigenvectors[2]
ev4 = sorted_eigenvectors[3]
ev5 = sorted_eigenvectors[4]
ev6 = sorted_eigenvectors[5]
```


```python
eigen_matrix = np.hstack((ev1.reshape(dims,1),
                          ev2.reshape(dims,1),
                          ev3.reshape(dims,1),
                          ev4.reshape(dims,1),
                          ev5.reshape(dims,1),
                          ev6.reshape(dims,1)))
```


```python
eigen_matrix[:3]
```




    array([[ 0.48931422, -0.11050274, -0.12330157, -0.22961737, -0.08261366,
             0.10147858],
           [-0.23858436,  0.27493048, -0.44996253,  0.07895978,  0.21873452,
             0.41144893],
           [ 0.46363166, -0.15179136,  0.23824707, -0.07941826, -0.05857268,
             0.06959338]])




```python
Y = X.dot(eigen_matrix).join(df_red_wine['quality'])
```


```python
Y.head()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.619530</td>
      <td>0.450950</td>
      <td>-1.774454</td>
      <td>0.043740</td>
      <td>0.067014</td>
      <td>-0.913921</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.799170</td>
      <td>1.856553</td>
      <td>-0.911690</td>
      <td>0.548066</td>
      <td>-0.018392</td>
      <td>0.929714</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.748479</td>
      <td>0.882039</td>
      <td>-1.171394</td>
      <td>0.411021</td>
      <td>-0.043531</td>
      <td>0.401473</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.357673</td>
      <td>-0.269976</td>
      <td>0.243489</td>
      <td>-0.928450</td>
      <td>-1.499149</td>
      <td>-0.131017</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.619530</td>
      <td>0.450950</td>
      <td>-1.774454</td>
      <td>0.043740</td>
      <td>0.067014</td>
      <td>-0.913921</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Distribution of PC against each other for Good(1) and Bad(0) quaity of wine


```python
sns.pairplot(data=Y, hue="quality", kind="scatter")
```




    <seaborn.axisgrid.PairGrid at 0x1edbed67688>




    
<img src="/images/p1/output_45_1.png">
    


## PCA with scikit-learn library


```python
from sklearn.decomposition import PCA
pca = PCA(n_components=0.85)
Y_sklearn = pca.fit_transform(X)
```


```python
plt.clf()
fig, ax = plt.subplots()

ax.plot(range(6), np.cumsum(pca.explained_variance_), '-o')

plt.xlabel('Number of Components')
plt.ylabel('Percent of Variance Explained')

plt.show()
```


    <Figure size 432x288 with 0 Axes>



    
<img src="/images/p1/output_48_1.png">
    



```python
np.cumsum(pca.explained_variance_)
```




    array([3.10107182, 5.02818671, 6.5797005 , 7.79369225, 8.75358463,
           9.41360567])




```python
Y_sklearn_plt = pd.DataFrame(Y_sklearn, 
                             index=Y.index, 
                             columns=Y.columns[:-1]).join(df_red_wine['quality'])
```


```python
sns.pairplot(data=Y_sklearn_plt, hue="quality")
```




    <seaborn.axisgrid.PairGrid at 0x1edc1a87608>




    
<img src="/images/p1/output_51_1.png">
    


<br>
<br>

## Performance of classfier : with and without PCA

### Without PCA


```python
y = df_red_wine['quality'].values
```


```python
# split dataset for training and testing, and use a logistic regression as classifier
X_train, X_test, y_train, y_test = train_test_split(df_red_wine.drop('quality', axis=1), y, test_size=0.25)
```


```python
classifier = LogisticRegression(random_state= 0)
```


```python
classifier.fit(X_train, y_train)
```




    LogisticRegression(random_state=0)




```python
y_pred = classifier.score(X_test, y_test)
y_pred
```




    0.7125



### With PCA


```python
X_train, X_test, y_train, y_test = train_test_split(Y_sklearn, y, test_size=0.3)
```


```python
classifier_with_pca = LogisticRegression(random_state=0)
classifier_with_pca.fit(X_train, y_train)
```




    LogisticRegression(random_state=0)




```python
y_pred = classifier_with_pca.score(X_test, y_test)
y_pred
```




    0.7458333333333333



## Resources

* https://www.sciencedirect.com/science/article/pii/B9780080448947013038 
* H. Abdi and L. J. Williams, "Principal component analysis," Wiley Interdisciplinary Reviews. Computational Statistics, vol. 2, (4), pp. 433-459, 2010.


