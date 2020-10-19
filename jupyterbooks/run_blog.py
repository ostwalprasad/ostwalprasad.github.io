
"""
ALWAYS RUN run.py AFTER COMPLETING EDITIONS IN run_blog.py
"""


import pathlib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# plot settings
sns.set_theme(style="whitegrid")

# create the output directory if it does not exist
# documentation here:
# https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir
pathlib.Path('./output').mkdir(exist_ok=True)

# *************************
# Question 1
# *************************

# Read data
df_Q1 = pd.read_csv('./specs/SensorData_question1.csv')

sns.boxplot(x="variable", y="value", data=pd.melt(df_Q1))
plt.ylabel('Input features')
plt.xlabel('Magnitude')
plt.show()

"""# Range of value dispersion [min, max] for attributes
# print(list(zip(list(round(df_Q1.min(axis = 0),3)),
#                list(round(df_Q1.max(axis = 0),3)))))
Result: [(1.129, 4.711), (0.825, 2.57), (2.706, 5.887), (2.278, 6.013), (0.084, 2.754), (-0.176, 2.38), 
         (-0.016, 1.266), (-0.027, 0.798), (0.042, 2.561), (0.867, 5.267), (0.704, 5.616), (4.686, 5.818)]
"""

# Preserve list of all input attributes SensorData_question1.csv
input_columns = list(df_Q1.columns)

# Preserve copy of attribute Input 3 and Input 12 within dataFrame
df_Q1['Original Input3'] = df_Q1['Input3']
df_Q1['Original Input12'] = df_Q1['Input12']

# Z-score transformation of attribute 'Input3'
"""
StandardScaler module in scikit-learn performs Z score transformation [(x-u)/s] ... u = mean, s = standard deviation 
documentation here: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# df_Q1['Input3'] = StandardScaler().fit_transform(df_Q1[['Input3']])
Output of this transform does not pass test case -
Error during test case : AssertionError: -1.107894096297504 != -1.11069 within 5 places
Hence, manual calculation is done
"""
input3 = df_Q1[['Input3']]
df_Q1['Input3'] = (input3 - input3.mean()) / input3.std()

# Min-Max transformation of attribute 'Input12'
# documentation:
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler
df_Q1['Input12'] = MinMaxScaler(feature_range=[0.0, 1.0]).fit_transform(df_Q1[['Input12']])

# calculate 'Average Input' --> mean of input attributes in SensorData_question1.csv
df_Q1['Average Input'] = df_Q1[input_columns].mean(axis=1, skipna=True)

""" Range of value dispersion [min, max] for attributes
# print(list(zip(list(round(df_Q1.min(axis = 0),3)),
#                list(round(df_Q1.max(axis = 0),3)))))
result:[(1.129, 4.711), (0.825, 2.57), (-1.506, 1.171), (2.278, 6.013), (0.084, 2.754), 
        (-0.176, 2.38), (-0.016, 1.266), (-0.027, 0.798), (0.042, 2.561), (0.867, 5.267),
        (0.704, 5.616), (0.0, 1.0), (2.706, 5.887), (4.686, 5.818), (0.594, 2.779)]
"""

# save the dataFrame to csv
# documentation here:
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
df_Q1.to_csv('./output/question1_out.csv', index=False, float_format='%g')

# *************************
# *************************
# Question 2
# *************************
# Read data
df_Q2 = pd.read_csv('./specs/DNAData_question2.csv')

# reduce dimensionality using Principle component analysis; so that at least 95% of variance is explained
pca_model = PCA(n_components=0.95)
np_Q2_PCA = pca_model.fit_transform(df_Q2)
df_Q2_PCA = pd.DataFrame(np_Q2_PCA)

"""
Preserved variance with all 22 components : 0.9520256  
print(pca_model.explained_variance_ratio_.cumsum())
Result: [0.36082343 0.51308479 0.61507636 0.67982246 0.72655958 0.76528613
         0.79751318 0.82139094 0.83841288 0.85486741 0.86862588 0.8814492
         0.89357807 0.90369637 0.91181695 0.91930295 0.92617387 0.9321854
         0.93794901 0.94284274 0.94760497 0.9520256 ]
"""

""" Range of value dispersion [min, max] for attributes
print(df_Q2_PCA.shape, list(zip(list(round(df_Q2_PCA.min(axis = 0),3)),
               list(round(df_Q2_PCA.max(axis = 0),3)))))
Result: [(-8.294, 30.6), (-10.724, 13.626), (-8.206, 12.503), (-4.751, 10.496), (-9.106, 9.725), (-4.99, 10.483), 
         (-7.247, 7.824), (-6.004, 6.79), (-3.866, 4.466), (-2.684, 8.61), (-2.988, 3.486), (-3.347, 5.714), 
         (-3.781, 4.113), (-3.837, 4.289), (-2.304, 5.644), (-2.689, 3.141), (-2.142, 3.415), (-2.022, 2.829), 
         (-1.822, 2.766), (-2.281, 2.763), (-1.901, 2.559), (-2.12, 2.64)]
"""

# Preserve list of all renamed PCA attributes
input_columns = list(df_Q2_PCA.columns)

# create bins by frequency [count of elements in bin]
# for each binned column 'X', store the bins in new column 'pcaX_width'
# create bins by width [range of values of elements in bin]
# for each binned column 'X', store the bins in new column 'pcaX_freq'
for col in input_columns:
    df_Q2_PCA["pca" + str(col) + "_width"] = pd.cut(df_Q2_PCA[col], 10)  # bin by width
    df_Q2_PCA["pca" + str(col) + "_freq"] = pd.qcut(df_Q2_PCA[col], 10)  # bin by frequency

# drop pca attributes once discretized into bins
df_Q2_PCA.drop(input_columns, axis=1, inplace=True)

# create final data set
df_Q2 = df_Q2.join(df_Q2_PCA)

# save the dataFrame to csv
# documentation here:
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html

df_Q2.to_csv('./output/question2_out.csv', index=False, float_format='%g')
