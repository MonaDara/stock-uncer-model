import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import tushare as ts
import pandas_datareader.data as web
#fetching data
#df = ts.get_h_data('002711',start = '2012-01-01')
style.use('ggplot')
#saving data in csv format
#df.to_csv('002711.csv')

#read the csv file and defining index or first col
df = pd.read_csv('C:/Users/Monica/Documents/Monica Files/report/workspace/Python/002195.csv', parse_dates= True, index_col=0)

print(df.head())
print(df.tail(6))
#print(df[['low','close']].head())
df['low'].plot()
plt.show()

# adding a new column like moving average
df['100ma'] = df['low'].rolling(window=100, min_periods=0).mean()
print(df.tail())

# for the printing "head" the first 100 rows don't have ma so we drop these rows.
#inplace means remove when it is true.
df.dropna(inplace=True)

print(df.head)

ax1 = plt.subplot2grid((6,1),(0,0),rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1),(5,0),rowspan=1, colspan=1, sharex=ax1)

ax1.plot(df.index, df['low'])
ax1.plot(df.index, df['100ma'])
#ax2.bar(df.index, df['date'])

plt.show()
