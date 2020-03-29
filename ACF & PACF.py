import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import tushare as ts
import pandas_datareader.data as web
from scipy import stats
import numpy as np
import matplotlib
import statsmodels.api as sm
from datetime import datetime
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import pandas_datareader.data as web
from lmfit.models import LinearModel, StepModel
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas import Series
from statsmodels.tsa.stattools import adfuller

#fetching data
#df = ts.get_hist_data('002384')
style.use('ggplot')

##saving data in csv format
#df.to_csv('C:/Users/Monica/Documents/Monica Files/report/CSV/Orig/600696.csv')

##read the csv file and defining index or first col
df = pd.read_csv('C:/Users/Monica/Documents/Monica Files/report/workspace/Python/002384.csv', parse_dates= True, index_col=0)

##Data Statistics
print(df.iloc[:,:4].head())
#print(df[['open','high','close','low']].describe())

##Plots
#df['low'].plot()
#plt.show()

##ACF and PACF plots for the differenced time series
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['low'].diff().dropna(), lags=90, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['low'].diff().dropna(), lags=90, ax=ax2)
plt.show()
