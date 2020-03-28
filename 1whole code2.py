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
#df = ts.get_hist_data('600696')
style.use('ggplot')

##saving data in csv format
#df.to_csv('C:/Users/Monica/Documents/Monica Files/report/CSV/Orig/600696.csv')

##read the csv file and defining index or first col
df = pd.read_csv('002184.csv', parse_dates= True, index_col=0)

#print("Data Statistics:")
#print(df.iloc[:,:4].head())
print(df[['open','high','close','low']].describe())
                                 
##Plots
#df['low'].plot()
#plt.show()
##ACF and PACF plots for the differenced time series
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['low'].diff().dropna(), lags=15, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['low'].diff().dropna(), lags=15, ax=ax2)
#plt.show()
##Measures of Central Tendency
#np.random.seed(1)
#data = np.round(np.random.normal(df['low']))
#plt.hist(data, bins=10, range=(3,60), edgecolor='black')
#df.hist(alpha=0.5, figsize=(16,10))
#plt.show

#print("Normality Check:")
#print("(Jarque-Bera, P-value):", stats.jarque_bera(df['low']))

#print("skewness:")
#print(df['low'].skew())

#print("Kurtosis:")
#print(df['low'].kurtosis())


#print("Stationary Check:")
#series = pd.read_csv('002711.csv')
#X = series.low.values
#result = adfuller(X)
#print('ADF Statistic: %f' % result[0]) #"The more negative this statistic, the more likely we are to reject the null hypothesis (we have a stationary dataset)."
##if our statistic value is less than the value at 1%, This suggests that we can reject the null hypothesis with a significance level of less than 1%

#print('p-value: %f' % result[1])
#print('Critical Values:')
#for key, value in result[4].items():
#	print('\t%s: %.3f' % (key, value))

##Bootstrap 2

##Reading the data 
df1 = pd.read_csv('002184.csv', low_memory=False)
df = df1.head(500)

print(df['low'].describe())
print("low Price statistics:")
print('Orig_mean:', df['low'].mean())
print('Orig_median:', df['low'].median())
print('Orig_quant10%:', df['low'].quantile(0.9))
print('Orig_max:', df['low'].max())
print('Orig_min:', df['low'].min())


# ecdf = empirical cumulative distribution function
## A function that takes as input a 1D array of data then returns the x and y values of the ECDF.
##ECDFs are among the most important plots in statistical analysis. 
def ecdf(df):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(df)

    # x-data for the sorting : x
    x = np.sort(df)

    # y-data of ECDF go from 1/n to 1 in equally spaced increment
    y = np.arange(1, n+1)/n

    return x, y


for _ in range(50):
    # Generate bootstrap sample: bs_sample
    bs_sample = np.random.choice(df['low'], size=len(df['low']))

    # Compute and plot ECDF from bootstrap sample
    x, y = ecdf(bs_sample)
    _ = plt.plot(x, y, marker='.', linestyle='none',
                 color='gray', alpha=0.1)
print('bootstrapped price statistics:')
print('bs2mean:', np.mean(bs_sample))
print('bs2median:', np.median(bs_sample))
print('bs2std:', np.std(bs_sample))

# Compute and plot ECDF from original data
x, y = ecdf(df['low'])
_ = plt.plot(x, y, marker='.')

# Make margins and label axes
plt.margins(0.02)
_ = plt.xlabel('Date')
_ = plt.ylabel('ECDF')

# Show the plot
plt.show()

#create dataframe for Xresampled
df_bssample = pd.DataFrame(data = bs_sample)
df_bssample.to_csv('002184_bs2.csv')
print(df_bssample.head())

#outlier removing,
low = 0
high = 1
quant_df = df_bssample.quantile([low, high])
print(quant_df.head())
filt_df = df_bssample.apply(lambda x: x[(x>quant_df.loc[low,x.name])& (x < quant_df.loc[high,x.name])], axis=0)
filt_df.to_csv('filt_bs2_002184.csv')
print(filt_df.head())
print('Filter Des:',filt_df.describe())

##Fitting model

##Read the csv file of bootstraped data
bsdf = pd.read_csv('filt_bs2_002184.csv')
##A glance on the data
bsdf.tail()
##data variation over resample
###df.plot.line(x=df.iloc[1:,0], y=df.iloc[1:,1])
print("bs2statistics:", bsdf.iloc[0:,1].describe())
print(bsdf.iloc[0:,1].head())
# Make margins and label axes
plt.margins(0.02)
_ = plt.xlabel('Resample No.')
_ = plt.ylabel('Price_Low')
###plt.show()

##Simple Exponential Smoothing is good for data with no trend and pattern
fit1 = SimpleExpSmoothing(bsdf.iloc[0:,1]).fit(smoothing_level=0.2,optimized=False)
fcast11 = fit1.forecast(6).rename(r'$\alpha=0.2$')
# plot
fcast11.plot(marker='o', color='blue', legend=True)
#fit1.fittedvalues.plot(marker='o',  color='blue')

fit2 = SimpleExpSmoothing(bsdf.iloc[0:,1]).fit(smoothing_level=0.6,optimized=False)
fcast12 = fit2.forecast(6).rename(r'$\alpha=0.6$')
# plot
fcast12.plot(marker='o', color='red', legend=True)
#fit2.fittedvalues.plot(marker='o', color='red')

fit3 = SimpleExpSmoothing(bsdf.iloc[0:,1]).fit()
fcast13 = fit3.forecast(6).rename(r'$\alpha=%s$'%fit3.model.params['smoothing_level'])
# plot
fcast13.plot(marker='o', color='green', legend=True)
#fit3.fittedvalues.plot(marker='o', color='green')

plt.margins(0.02)
_ = plt.xlabel('Days')
_ = plt.ylabel('Forecast_Price_Low_Exp Smooth')
print('SES result1:',fcast11, sep=",")
print('SES result2:',fcast12, sep = ",")
print('SES result3:',fcast13, sep = ",")
plt.show()

fcast1 = pd.concat([fcast11, fcast12, fcast13], axis=1)
fcast1.to_csv('fcast1.csv')

##Holt
fit1 = Holt(bsdf.iloc[0:,1]).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
fcast21 = fit1.forecast(6).rename("Holt's linear trend")
fit2 = Holt(bsdf.iloc[0:,1], exponential=True).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
fcast22 = fit2.forecast(6).rename("Exponential trend")
fit3 = Holt(bsdf.iloc[0:,1], damped=True).fit(smoothing_level=0.8, smoothing_slope=0.2)
fcast23 = fit3.forecast(6).rename("Additive damped trend")

#fit1.fittedvalues.plot(marker="o", color='blue')
fcast21.plot(color='blue', marker="o", legend=True)
#fit2.fittedvalues.plot(marker="o", color='red')
fcast22.plot(color='red', marker="o", legend=True)
#fit3.fittedvalues.plot(marker="o", color='green')
fcast23.plot(color='green', marker="o", legend=True)
plt.margins(0.02)
_ = plt.xlabel('Days')
_ = plt.ylabel('Forecast_Price_Low_Holt')
print('Holt result1:',fcast21 ,sep = ",")
print('Holt result2:',fcast22 ,sep = ",")
print('Holt result3:',fcast23 ,sep = ",")
plt.show()
fcast2 = pd.concat([fcast21, fcast22, fcast23], axis=1)
fcast2.to_csv('fcast2.csv')

##Holt-Winters' Method
fit1 = ExponentialSmoothing(bsdf.iloc[0:,1], seasonal_periods=4, trend='add', seasonal='add').fit(use_boxcox=True)
fit2 = ExponentialSmoothing(bsdf.iloc[0:,1], seasonal_periods=4, trend='add', seasonal='mul').fit(use_boxcox=True)
fit3 = ExponentialSmoothing(bsdf.iloc[0:,1], seasonal_periods=4, trend='add', seasonal='add', damped=True).fit(use_boxcox=True)
fit4 = ExponentialSmoothing(bsdf.iloc[0:,1], seasonal_periods=4, trend='add', seasonal='mul', damped=True).fit(use_boxcox=True)
fcast31 = fit1.forecast(6).rename("HW1")
fcast32 = fit2.forecast(6).rename("HW2")
fcast33 = fit3.forecast(6).rename("HW3")
fcast34 = fit1.forecast(6).rename("HW4")
print('HW result1:',fcast31 ,sep = ",")
print('HW result2:',fcast32 ,sep = ",")
print('HW result3:',fcast33 ,sep = ",")
print('HW result4:',fcast34 ,sep = ",")
#fit1.fittedvalues.plot(style='--', color='red')
#fit2.fittedvalues.plot(style='--', color='green')
fit1.forecast(6).plot(style='--', marker="o", color='red', legend=False)
fit2.forecast(6).plot(style='--', marker='o', color='green', legend=False)
fit3.forecast(6).plot(style='--', marker='o', color='yellow', legend=False)
fit4.forecast(6).plot(style='--', marker='o', color='purple', legend=False)
plt.margins(0.02)
_ = plt.xlabel('Days')
_ = plt.ylabel('LPF_Holt-Winters')

plt.show()

print("Price Forecasting using Holt-Winters method with both additive and multiplicative seasonality.")


fcast3 = pd.concat([fcast31, fcast32, fcast33,fcast34], axis=1)
fcast3.to_csv('fcast3.csv')


