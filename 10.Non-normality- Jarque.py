from datetime import datetime
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates 
from matplotlib import style
import matplotlib.dates as mdates
import pandas as pd
import tushare as ts
import pandas_datareader.data as web
import numpy as np
import statsmodels.api as sm
from lmfit.models import LinearModel, StepModel
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera as JB
##Read the csv file 
df = pd.read_csv('C:/Users/Monica/Documents/Monica Files/report/CSV/002028_bssample.csv')
df.index=pd.to_datetime(df['date'])
df['low'].plot()
plt.show



 
#H0: The Data Are Normally Distributed
#Ha: The Data Are Not Normally Distributed
