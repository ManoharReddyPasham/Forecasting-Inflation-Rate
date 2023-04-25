#!/usr/bin/env python
# coding: utf-8

# In[600]:


import pandas as pd
get_ipython().system('pip install openpyxl')


# In[601]:


df=pd.read_excel('file:///Users/dilipkumarallu/Desktop/PROJECT_DATASET.xlsx')


# In[602]:


df=pd.DataFrame(df)


# In[603]:


df=df.dropna(axis='columns')


# In[604]:


df.to_csv('inf.csv')


# In[605]:


df = pd.read_csv('inf.csv')


# In[606]:


df.head()


# In[607]:


df=df.dropna(axis='columns')


# In[608]:


get_ipython().system('pip install seaborn')
import seaborn as sns


# In[609]:


from sklearn import linear_model
import matplotlib.pyplot as plt
x = df['YEAR']
y = df['Inflation rate, end of period consumer prices (Annual percent change)']
plt.scatter(x,y,c='r')
plt.xlabel('YEAR')
plt.ylabel("Inflation rate")
plt.title('SCATTERPLOT')


# In[610]:


x = df['YEAR']
y = df['Inflation rate, end of period consumer prices (Annual percent change)']
plt.plot(x,y,c='r')
plt.xlabel('YEAR')
plt.ylabel("Inflation rate")
plt.title('LINEAR PLOT')


# In[611]:


x = df['YEAR']
y = df['Percapita GDP']
plt.plot(x,y,c='r')
plt.xlabel('YEAR')
plt.ylabel("percapita GDP")
plt.title('LINEAR PLOT')


# In[612]:


labels = 'Inflation rate, end of period consumer prices (Annual percent change)','Percapita GDP', 'Population(in millions)','Purchasing Power'
sizes = 200,170,300,500
explode=(0.1,0.2,0,0)

plt.pie(labels = labels,explode = explode,x=sizes ,shadow=False)


# In[613]:


df.corr()


# In[614]:



sns.heatmap(df.corr())


# In[615]:


sns.pairplot(df)


# In[616]:


sns.pairplot(df,hue = 'YEAR')


# In[617]:


sns.distplot(df['Inflation rate, end of period consumer prices (Annual percent change)'])


# # PREDICTION

# In[618]:


X = df[['YEAR', 'Purchasing Power', 'Population(in millions)', 'Current account balance U.S. dollars (Billions of U.S. dollars)','GDP', 'Percapita GDP']]
y = df['Inflation rate, end of period consumer prices (Annual percent change)']


# In[619]:


from sklearn.model_selection import train_test_split


# In[620]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.096, random_state = 0)


# In[621]:


X


# In[622]:


regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)


# In[623]:


predictions = regr.predict(x_test)


# In[624]:


plt.scatter(y_test, predictions)


# In[625]:


regr.coef_
regr.score(x_test, y_test)


# In[626]:


# predict the y values
y_pred=regr.predict(x_test)
# a data frame with actual and predicted values of y
evaluate = pd.DataFrame({"Predicted": y_pred.flatten()})
evaluate.head()


# In[627]:


# plt.plot(x_train['YEAR'],y_train, color = "black")
plt.plot(X['YEAR'],y, color = "red")
plt.plot(x_test['YEAR'],evaluate['Predicted'], color='green', label = 'Predictions')
plt.ylabel('Inflation rate')
plt.xlabel('YEAR')
plt.xticks(rotation=45)
plt.title("Train/Test split for Inflation Data")
plt.show()


# In[628]:


evaluate.head(10).plot(kind = "bar")


# In[571]:


X = df[['YEAR', 'Purchasing Power', 'Population(in millions)', 'Current account balance U.S. dollars (Billions of U.S. dollars)', 'Inflation rate, end of period consumer prices (Annual percent change)']]
y = df['GDP']


# In[572]:


x_train, x_test1, y_train, y_test = train_test_split(X, y, test_size = 0.096, random_state = 0)


# In[573]:


regr2 = linear_model.LinearRegression()
regr2.fit(x_train, y_train)


# In[574]:


regr2.coef_
regr2.score(x_test1,y_test)


# In[575]:


# predict the y values
y_pred=regr2.predict(x_test1)
# a data frame with actual and predicted values of y
evaluate2 = pd.DataFrame({"Actual": y_test.values.flatten(), "Predicted": y_pred.flatten()})
evaluate2.head()


# In[576]:


# plt.plot(x_train['YEAR'],y_train, color = "black")
plt.plot(X['YEAR'],y, color = "red")
plt.plot(x_test1['YEAR'],evaluate2['Predicted'], color='green', label = 'Predictions')
plt.ylabel('Per capita GDP')
plt.xlabel('YEAR')
plt.xticks(rotation=45)
plt.title("Train/Test split for GDP Data")
plt.show()


# In[577]:


evaluate.head(10).plot(kind = "bar")


# In[579]:


plt.plot(x_test1['YEAR'],evaluate2['Predicted'], color='green', label = 'Predictions')
plt.ylabel('Inflation rate')
plt.xlabel('YEAR')
plt.xticks(rotation=45)
plt.title("Predictions for GDP Data")
plt.show()


# In[580]:


get_ipython().system('pip install pandas-datareader')


# In[581]:


get_ipython().system('pip install statsmodels')
from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[582]:


ARMAmodel = SARIMAX(y_train, order = (1, 0, 1))


# In[583]:


ARMAmodel = ARMAmodel.fit()


# In[584]:


y_pred = ARMAmodel.get_forecast(len(y_test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = ARMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = y_test.index
y_pred_out = y_pred_df["Predictions"] 


# In[585]:


plt.plot(y_pred_out, color='green', label = 'Predictions')
plt.legend()


# In[586]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.ylabel('Inflation rate')
plt.xlabel('Date')
plt.xticks(rotation=45)


# In[587]:


plt.plot(df['YEAR'], df['Inflation rate, end of period consumer prices (Annual percent change)'], )


# In[588]:


train = df[df['YEAR']<2010]
test = df[df['YEAR']>2009]


# In[589]:


plt.plot(train['YEAR'],train['Inflation rate, end of period consumer prices (Annual percent change)'], color = "black")
plt.plot(test['YEAR'],test['Inflation rate, end of period consumer prices (Annual percent change)'], color = "red")
plt.ylabel('Inflation rate')
plt.xlabel('YEAR')
plt.xticks(rotation=45)
plt.title("Train/Test split for Inflation Data")
plt.show()


# In[590]:


from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[591]:


y = train['Inflation rate, end of period consumer prices (Annual percent change)']


# In[592]:


ARMAmodel = SARIMAX(y, order = (1, 0, 1))


# In[593]:


ARMAmodel = ARMAmodel.fit()


# In[594]:


test


# In[595]:


y_pred = ARMAmodel.get_forecast(len(test['YEAR']))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = ARMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df['YEAR'] = test['YEAR']
y_pred_out = y_pred_df["Predictions"] 


# In[596]:


print(y_pred_out)


# In[597]:


plt.plot(train['YEAR'],train['Inflation rate, end of period consumer prices (Annual percent change)'], color = "black")
plt.plot(test['YEAR'],test['Inflation rate, end of period consumer prices (Annual percent change)'], color = "red")
plt.plot(y_pred_df['YEAR'],y_pred_df["Predictions"], color='green', label = 'Predictions')
plt.ylabel('Inflation rate')
plt.xlabel('YEAR')
plt.xticks(rotation=45)
plt.title("Train/Test split for Inflation Data")
plt.show()

