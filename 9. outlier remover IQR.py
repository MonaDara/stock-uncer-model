import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import tushare as ts
import pandas_datareader.data as web
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib

#load the packages
MAX_ROWS = 10
pd.set_option('display.max_rows', MAX_ROWS)
pd.set_option('display.max_columns', 2)
sns.set_style("whitegrid")
sns.set_context("paper")

#fetching data
#df = ts.get_h_data('002028',start = '2015-01-01')
style.use('ggplot')
#saving data in csv format
#df.to_csv('002028.csv')
#read the csv file and defining index or first col
df = pd.read_csv('600696.csv')
#print(df.head())
##print(df.tail(6))
#print(df[['low','close']].head())
df

#visualize data
##setting i as ‘low’. I do this sometimes to save me having to changeit in
##many places if I want to view other variables. We are also setting
##the x axis to a min and max values based on the min and max of the variable.
i = 'low'

plt.figure(figsize=(10,8))
plt.subplot(211)
plt.xlim(df[i].min(), df[i].max()*1.1)

ax = df[i].plot(kind='kde')

plt.subplot(212)
plt.xlim(df[i].min(), df[i].max()*1.1)
sns.boxplot(x=df[i])
plt.show()
print(df[i])
#df[i].to_csv('dfi.csv')
#Transform the data
##We set all zero values in low to NaN. A zero can cause a problem when using
##a log transform
##We drop all the rows with a NaN,(change df.dropna(inplace=True) to
##df.dropna(subset=[‘low’], inplace=True) to keep more data)
##We create a new variable called ‘Log_’ + i where i is ‘low’,
##so the new variable is Log_low

# Remove any zeros (otherwise we get (-inf)
df.loc[df.low == 0, 'low'] = np.nan
 
# Drop NA
df.low.dropna(inplace=True)
 
# Log Transform
#df['Log_' + i] = np.log(df[i])

##visualize data again
#i = 'Log_low'

plt.figure(figsize=(10,8))
plt.subplot(211)
plt.xlim(df[i].min(), df[i].max()*1.1)

ax = df[i].plot(kind='kde')

plt.subplot(212)
plt.xlim(df[i].min(), df[i].max()*1.1)
sns.boxplot(x=df[i])
plt.show()


#Determine the Min and Max cuttoffs for detecting the outliers
##Step 1, get the  Interquartile Range
##Step 2, calculate the upper and lower values

q75, q25 = np.percentile(df.low.dropna(), [75 ,25])
iqr = q75 - q25

min = q25 - (iqr*1.5)
max = q75 +(iqr*1.5)
print(max)
#visualise this using similar code as shown above by adding plt.axvline.
#i = 'Log_low'

plt.figure(figsize=(10,8))
plt.subplot(211)
plt.xlim(df[i].min(), df[i].max()*1.1)
plt.axvline(x=min)
plt.axvline(x=max)

ax = df[i].plot(kind='kde')

plt.subplot(212)
plt.xlim(df[i].min(), df[i].max()*1.1)
sns.boxplot(x=df[i])
plt.axvline(x=min)
plt.axvline(x=max)
plt.show()

#identify the outliers
##First we set a new variable in the dataframe called ‘Outlier’ defaulted to 0,
##then is a row is outside this range we set it to 1. Note:
##i should still be ‘Log_low’
df['Outlier'] = 0

df.loc[df[i] < min, 'Outlier'] = 1
df.loc[df[i] > max, 'Outlier'] = 1
print(df[i])
#df[i].to_csv('dfi1.csv')
#Now we can plot the original data and the data without the outliers
#in (Clean Data).
#i = 'Log_low'

plt.figure(figsize=(10,8))
plt.subplot(211)
plt.xlim(df[i].min(), df[i].max()*1.1)

ax = df[i].plot(kind='kde')

plt.subplot(212)
plt.xlim(df[i].min(), df[i].max()*1.1)
sns.boxplot(x=df[i])
plt.show()




