---
title: 'Must have code snippets for every Data Science Notebook'
date: 2020-07-01 12:23:12 Z
categories:
- machine-learning
tags:
- machine learning
- python
- scikit-learn
- jupyter notebook

layout: post
comments: true
description: 'Must have code snippets for every Data Science Notebook'
author: Prasad Ostwal
---

I've been using Jupyter notebooks for quite a while and everytime I create a new notebook I have to write same 10-15 lines of *bare minimum* code with some visualization snippets that are mostly needed, so why not write them at once and use everytime? 

I always keep this page open in web browser to quickly copy snippets while *pythoning*.

So here are few code snippets..!

## Must have Imports


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
```

## Datasets

##### 1. Boston Housing Dataset


```python
from sklearn.datasets import load_boston
import pandas as pd

boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
print(f"Loaded {data.shape[0]} rows and {data.shape[1]} features.")
data['price'] = boston.target
y = data['price']
x = data.drop('price',axis=1)
data.head(5)
```

#### 2. Californa Housing Dataset


```python
from sklearn.datasets import fetch_california_housing
dataset = fetch_california_housing()
data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
print(f"Loaded {data.shape[0]} rows and {data.shape[1]} features.")
data['price'] = dataset.target
y = data['price']
x = data.drop('price',axis=1)
data.head(5)
```

#### 3. Iris Dataset


```python
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
print(f"Loaded {data.shape[0]} rows and {data.shape[1]} features.")
data['target'] = iris.target
y = data['target']
x = data.drop('target',axis=1)
data.head(5)
```

## Reading FIles

#### 2. Read CSV


```python
path = ""
file = ""
data = pd.read_csv(path+file)
data.shape
```

#### 2. Read Excel


```python
path = r""
file = ""
sheet_name=""
data = pd.read_excel(path+file,sheet_name=sheet_name)
data.shape
```

#### 2. Read JSON


```python
import json

with open('filename.json') as f:
    d = json.load(f)
d
```

#### 4. Read Text file into lines


```python
with open("file.txt") as file_in:
    lines = []
    for line in file_in:
        lines.append(line)
```

## EDA

#### 1. Correlation Plot


```python
def corrplot(df,save=False,title=None):
  """
  Plots correlation heatmap using Seaborn

  args:
  -----
  df: Pandas Dataframe
  save: True saves image
  title:Optional title for plot

  """
  cov = df.corr()
  plt.figure(figsize = (len(df.columns.to_list())*1,len(df.columns.to_list())*0.75))
  cols =  df.columns.to_list()
  ax = plt.axes()
  sns.heatmap(cov,annot=True,cmap="PiYG",yticklabels=cols,xticklabels=df.columns.to_list(),ax=ax)
  if title:
    ax.set_title(title)
  if save == True:
    plt.savefig("Heatmap.png")

corrplot(data,save=True,title="My Heatmap")
```

#### 2. KDE Plots


```python
def kde_plots(df,save=False):
  """
  Plots KDE plots of all features in dataframe using Seaborn

  args:
  -----
  df: Pandas Dataframe
  save: True saves image
  title:Optional title for plot

  """
  rows=math.ceil(len(df.columns)/4)
  fig, ax = plt.subplots(ncols=4,nrows=rows,figsize=(12, rows*2))
  ix=0
  fig.tight_layout()
  for row in ax:
      for col in row:
          sns.distplot(df.iloc[:, ix].dropna(),norm_hist=False,ax=col,label="")
          col.set_title(df.columns[ix]) 
          col.set_xlabel('')
          plt.text(0.2, 0.8,f'u:{round(df.iloc[:, ix].mean(),2)}\nsd={round(df.iloc[:, ix].std(),2)}', ha='center', va='center', transform=col.transAxes)
          ix+=1
          if ix == len(df.columns):
            break
  plt.subplots_adjust(hspace = 0.4)
  
  if save == True:
    plt.savefig("KDE_plots.png")

kde_plots(data,save=True)
```

I will add more snippets as an when I feel they are being used repetitively.

## Source

Download Jupyter notebook from [GitHub](https://github.com/ostwalprasad/ostwalprasad.github.io/blob/master/jupyterbooks/2020-07-01-Must have code snippets for every Data Science Notebook )




