#!/usr/bin/env python
# coding: utf-8

# # problem statement
# The objective is to analyze the flight booking dataset obtained from a platform which is used to book flight tickets. 
# A thorough study of the data will aid in the discovery of valuable insights that will be of enormous value to passengers. 
# Apply EDA, statistical methods and Machine learning algorithms in order to get meaningful information from it. 

# # Dataset Information
# Attributes--- Description
# Airline--- Name of the airline company
# Flight--- Plane's flight code
# Source City--- City from which the flight takes off
# Departure Time--- Time of Departure
# Stops--- Number of stops between the source and
#          destination cities
# Arrival--- Time Time of Arrival
# Destination City--- City where the flight will land
# Class--- Contains information on seat class
# Duration--- Overall amount of time taken to travel
#             between cities in hours.
# Days left--- Subtracting the trip date by the booking
#              date.
# Price--- Ticket price

# In[15]:


# Importing libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:


# Loading the data
df = pd.read_csv('Flight_Booking.csv')
df


# In[17]:


df.shape


# In[18]:


df.info()


# In[19]:


df.describe()


# In[20]:


# Checking missing values
df.isnull().sum()


# In[21]:


# Data Visualization
plt.figure(figsize=(15,5))
sns.lineplot(x=df['airline'],y=df['price'])
plt.title('Airlines vs Price',fontsize=15)
plt.xlabel('airline',fontsize=15)
plt.ylabel('price',fontsize=15)
plt.show()


# In[22]:


plt.figure(figsize=(15,5))
sns.lineplot(data=df,x='days_left',y='price',color='blue')
plt.title('Days_left for Departure Versus Ticket Price',fontsize=15)
plt.xlabel('Days_left for Departure',fontsize=15)
plt.ylabel('Price',fontsize=15)
plt.show()


# In[23]:


plt.figure(figsize=(10,5));
sns.barplot(x = 'airline',y ='price', data=df )


# In[24]:


plt.figure(figsize=(10,8));
sns.barplot(x ='class',y ='price',data=df, hue='airline')


# In[25]:


# Range of price of flights with source and destination city according to the days left


# In[26]:


fig,ax=plt.subplots(1,2,figsize=(20,6))
sns.lineplot(x='days_left',y='price',data=df,hue='source_city',ax=ax[0])
sns.lineplot(x='days_left',y='price',data=df,hue='destination_city',ax=ax[1])
plt.show()


# In[27]:


# Visualization of categorical features with countplot
plt.figure(figsize=(15,23))

plt.subplot(4, 2, 1)
sns.countplot(x=df["airline"], data=df)
plt.title("Frequency of Airline")

plt.subplot(4, 2, 2)
sns.countplot(x=df["source_city"], data=df)
plt.title("Frequency of Source_City")

plt.subplot(4, 2, 3)
sns.countplot(x=df["departure_time"], data=df)
plt.title("Frequency of Departure Time")

plt.subplot(4, 2, 4)
sns.countplot(x=df["stops"], data=df)
plt.title("Frequency of Stops")

plt.subplot(4, 2, 5)
sns.countplot(x=df["arrival_time"], data=df)
plt.title("Frequency of Arrival Time")

plt.subplot(4, 2, 6)
sns.countplot(x=df["destination_city"], data=df)
plt.title("Frequency of Destination City")

plt.subplot(4, 2, 7)
sns.countplot(x=df["class"], data=df)
plt.title("Class Frequency")

plt.show()


# # Label Encoding

# In[28]:


# Performing One Hot Encoding for categorical features of a dataframe
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["airline"]=le.fit_transform(df["airline"])
df["source_city"]=le.fit_transform(df["source_city"])
df["departure_time"]=le.fit_transform(df["departure_time"])
df["stops"]=le.fit_transform(df["stops"])
df["arrival_time"]=le.fit_transform(df["arrival_time"])
df["destination_city"]=le.fit_transform(df["destination_city"])
df["class"]=le.fit_transform(df["class"])
df.info()


# In[29]:


# Feature Selection
# Plotting the correlation graph to see the correlation between features and dependent variable.
plt.figure(figsize=(10,5))
sns.heatmap(df.corr(),annot=True,cmap="coolwarm")
plt.show()


# In[30]:


# Feature Selection
# Selecting the features using VIF. VIF should be less than 5. So drop the stops feature.
from statsmodels.stats.outliers_influence import variance_inflation_factor
col_list = []
for col in df.columns:
    if((df[col].dtype != 'object') & (col != 'price')):
        col_list.append(col)
        
X = df[col_list]
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                          for i in range(len(X.columns))]
print(vif_data)


# In[31]:


# Feature Selection
#Dropping the stops column. All features are having VIF less than 5. 
df=df.drop(columns=["stops"])

from statsmodels.stats.outliers_influence import variance_inflation_factor
col_list = []
for col in df.columns:
    if((df[col].dtype != 'object') & (col != 'price') ):
        col_list.append(col)
        
X = df[col_list]
vif_data = pd.DataFrame()
vif_data["vif"]=[variance_inflation_factor(X.values,i) 
                         for i in range(len(X.columns))]
print(vif_data)


# In[42]:


# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Assume df is your DataFrame containing the dataset

# Select numeric columns for scaling
numeric_columns = ['duration', 'days_left']  # Adjust as needed

# Select non-numeric columns for encoding
categorical_columns = ['flight']  # Adjust as needed

# Encode non-numeric columns (using label encoding in this example)
encoder = LabelEncoder()
df['flight_encoded'] = encoder.fit_transform(df['flight'])  # Assuming 'flight' is the column to be encoded

# Combine the numeric and encoded categorical columns
X = df[numeric_columns + ['flight_encoded']]

# Scale the numeric and encoded columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
y = df['price']
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create and fit the Linear Regression model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Predict using the model
y_pred = lr.predict(x_test)

# Create a DataFrame for the differences between actual and predicted values
difference = pd.DataFrame(np.c_[y_test, y_pred], columns=["Actual_value", "Predicted_Value"])

# Print or use the 'difference' DataFrame as needed
print(difference)


# In[44]:


# Linear regression
# Calculating r2 score,MAE, MAPE, MSE, RMSE. Root Mean square error(RMSE) of the Linear regression model is 7259.93 and Mean absolute percentage error(MAPE) is 34 percent. Lower the RMSE and MAPE better the model.
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
from sklearn import metrics
mean_abs_error=metrics.mean_absolute_error(y_test,y_pred)
mean_abs_error
from sklearn.metrics import mean_absolute_percentage_error
mean_absolute_percentage_error(y_test,y_pred)
mean_sq_error=metrics.mean_squared_error(y_test,y_pred)
mean_sq_error
root_mean_sq_error = np.sqrt(metrics.mean_squared_error(y_test,y_pred))
root_mean_sq_error


# In[ ]:





# In[45]:


# Linear regression
# Plotting the graph of actual and predicted price of flight 
sns.distplot(y_test,label="Actual")
sns.distplot(y_pred,label="Predicted")
plt.legend()


# In[46]:


# Decision Tree Regressor
# Mean absolute percentage error is 7.7 percent and RMSE is 3620 which is less than the linear regression mode

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
r2_score(y_test,y_pred)
mean_abs_error=metrics.mean_absolute_error(y_test,y_pred)
mean_abs_error
from sklearn.metrics import mean_absolute_percentage_error
mean_absolute_percentage_error(y_test,y_pred)
mean_sq_error
root_mean_sq_error=np.sqrt(metrics.mean_squared_error(y_test,y_pred))
root_mean_sq_error


# In[ ]:


# Random Forest Regressor
# Mean absolute percentage error is 7.3 percent and RMSE is 2824 which is less than the linear regression and decision tree model

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(x_train,y_train)
y_pred=rfr.predict(x_test)
r2_score(y_test,y_pred)
mean_abs_error=metrics.mean_absolute_error(y_test,y_pred)
mean_abs_error
from sklearn.metrics import mean_absolute_percentage_error
mean_absolute_percentage_error(y_test,y_pred)
mean_sq_error=metrics.mean_squared_error(y_test,y_pred)
mean_sq_error
root_mean_sq_error = np.sqrt(metrics.mean_squared_error(y_test,y_Pred))
root_mean_sq_error


# In[ ]:


sns.distplot(y_test,label="Actual")
sns.distplot(y_pred,label="Predicted")
plt.legend()


# In[ ]:





# In[ ]:





# In[ ]:




