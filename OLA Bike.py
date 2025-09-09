import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df=pd.read_csv('C:/Users/Pc solution & Laptop/Downloads/ola.csv')
df.info() #alot of null which ill take care of soon

parts=df['datetime'].str.split(" ",n=2,expand=True) #expand=True means “put the split parts into separate DataFrame columns instead of lists inside one column.
df['date']=parts[0]
df['time']=parts[1].str[:2].astype('int') #.str[:2] This slices the first two characters of the string.

parts=df['date'].str.split("-",n=3,expand=True)
df['year']=parts[0].astype('int')
df['month']=parts[1].astype('int')
df['day']=parts[2].astype('int')

df.describe()

numeric_cols=df.select_dtypes(include=np.number).columns.tolist()

imputer = SimpleImputer(strategy='median')
df['temp']=imputer.fit_transform(df[['temp']])

mask=df['humidity'].isnull()
target=df.loc[~mask,'humidity'] #gives non - null values which will 9k smth
inputs= df.loc[~mask,['season', 'weather', 'temp', 'time', 'year', 'month', 'day']] #gives rows where humidity is not null

px.scatter(df.sample(2000),
           title='Humidity Vs Temp.',
           x='humidity',
           y='temp'
           )
#plt.show()

px.scatter(df.sample(2000),
           title='Humidity Vs Temp.',
           x='humidity',
           y='temp',
           color='weather'
           )
#plt.show()

regressor=RandomForestRegressor(n_estimators=200,random_state=0)
regressor.fit(inputs,target)

y_pred = regressor.predict(inputs)

# Calculate metrics
r2 = r2_score(target, y_pred)
mae = mean_absolute_error(target, y_pred)

#print("R² Score:", r2)
#print("Mean Absolute Error:", mae)

# Select rows where humidity is missing
X_missing = df.loc[mask, ['season', 'weather', 'temp', 'time', 'year', 'month', 'day']]
preds=regressor.predict(X_missing)

df.loc[mask, 'humidity'] = preds

px.scatter(df.sample(2000),
           title='Windspeed Vs Temp.',
           x='windspeed',
           y='temp'
           )
#plt.show()

px.scatter(df.sample(2000),
           title='Windspeed Vs Temp.',
           x='windspeed',
           y='humidity'
           )
#plt.show()

plt.figure(figsize=(16, 10))
influence=df.corr(numeric_only=True)
sns.heatmap(influence, annot=True, cmap='Greens')
#plt.show()

mask=df['windspeed'].isnull()
target=df.loc[~mask,'windspeed']
inputs=df.loc[~mask,['temp', 'time', 'year', 'month', 'day','humidity']]

regressor=RandomForestRegressor(n_estimators=200,random_state=0)
regressor.fit(inputs,target)

y_pred = regressor.predict(inputs)

# Calculate metrics
r2 = r2_score(target, y_pred)
mae = mean_absolute_error(target, y_pred)

#print("R² Score:", r2)
#print("Mean Absolute Error:", mae)

X_missing=df.loc[mask,['temp', 'time', 'year', 'month', 'day','humidity']]

preds=regressor.predict(X_missing)
df.loc[mask,'windspeed']=preds

import datetime

df['datetime']=pd.to_datetime(df['datetime'])

df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day

def weekend_or_weekday(year,month,day):
    try:
        dt=datetime.date(year,month,day)
        return 0 if dt.weekday()<=4 else 1
    except ValueError:
        return np.nan

df['weekday']=df.apply(lambda x: weekend_or_weekday(x['year'],x['month'],x['day']),axis=1)

def am_or_pm(time):
    if time<12:
        return 0
    else:
        return 1
df['am_or_pm']=df.apply(lambda x: am_or_pm(x['time']),axis=1)
#df['am_or_pm']=df['time'].apply(am_or_pm) same thing

import holidays

def is_holiday(x):
    pak_holidays=holidays.country_holidays('PK')
    if pak_holidays.get(x):
        return 1
    else:
        return 0
df['holidays']=df['date'].apply(is_holiday)

# Helps capture non-linear effects that trees might miss otherwise.

df['temp_weather'] = df['temp'] * df['weather']

df['holiday_hour'] = df['holidays'] * df['time']

sns.lineplot(x="day", y="count", data=df)
#plt.show()

sns.lineplot(x="time", y="count", data=df)
#plt.show()

sns.lineplot(x="month", y="count", data=df)
#lt.show()

df.info()# no null values left

fig, axs = plt.subplots(3, 2, figsize=(12, 12))

axs = axs.flatten()

features = ['season', 'weather', 'holidays', 'am_or_pm', 'year', 'weekday']

for i, feature in enumerate(features):
    sns.barplot(x=feature, y='count', data=df, ax=axs[i])
    axs[i].set_title(f'Count vs {feature}')

plt.tight_layout()
#plt.show()

sns.histplot(x='temp', data=df, bins=50, color="skyblue", edgecolor="black")
#plt.show()

sns.histplot(x='windspeed', data=df, bins=50, color="skyblue", edgecolor="black")
#plt.show()

px.box(df, x='temp')
#plt.show() #alot of outliers

px.box(df, x='windspeed')
#plt.show() # alot of outliers here too

#finding outliers using inter quartile range, and checking if they're actual values or errors.
def find_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR
    return series[(series < lower_bound) | (series > upper_bound)]

wind_outliers = find_outliers_iqr(df['windspeed'])
temp_outliers = find_outliers_iqr(df['temp'])

#print("Windspeed outliers:\n", wind_outliers)
#print("Temperature outliers:\n", temp_outliers)

#In this area, this is the maximum windspeed and temprature so anything out of this range is an error.
df.loc[df['windspeed']>60,'windspeed']=np.nan
df.loc[(df['temp']>50) | (df['temp']<-30),'temp']=np.nan

imputer=SimpleImputer(strategy='mean')
df['temp']=imputer.fit_transform(df[['temp']])
df['windspeed']=imputer.fit_transform(df[['windspeed']])

px.box(df, x='windspeed')
#plt.show()

px.box(df, x='temp')
#plt.show()

#Using either casual or registered as a feature is cheating, because you’re basically giving the model a piece of the answer.
#Example: If registered = 30 and casual = 20, the model doesn’t have to learn anything—it can just add them to predict count = 50.
#This will result in perfect correlation and unrealistic model performance, but it won’t generalize to real predictions where you don’t know registered or casual in advance.

df.drop(['registered','casual'], axis=1, inplace=True)

train_val_df,test_df=train_test_split(df, test_size=0.2, random_state=42)
train_df,val_df=train_test_split(train_val_df, test_size=0.25, random_state=42)

train_target=train_df['count']
val_target=val_df['count']
test_target=test_df['count']


train_df=train_df.drop(['count','datetime','date'], axis=1)
val_df=val_df.drop(['count','datetime','date'], axis=1)
test_df=test_df.drop(['count','datetime','date'], axis=1)

#Even though time is numeric (0–23), it’s circular: 23 → 0. Using sin/cos helps the model capture that pattern
train_df['time_sin'] = np.sin(2 * np.pi * train_df['time'] / 24)
train_df['time_cos'] = np.cos(2 * np.pi * train_df['time'] / 24)

val_df['time_sin'] = np.sin(2 * np.pi * val_df['time'] / 24)
val_df['time_cos'] = np.cos(2 * np.pi * val_df['time'] / 24)

test_df['time_sin'] = np.sin(2 * np.pi * test_df['time'] / 24)
test_df['time_cos'] = np.cos(2 * np.pi * test_df['time'] / 24)

model=RandomForestRegressor(
    n_estimators=500,        # number of trees
    max_depth=20,            # max depth of each tree
    min_samples_split=5,     # min samples to split a node
    min_samples_leaf=2,      # min samples in a leaf
    max_features='sqrt',     # number of features to consider at each split
    random_state=42,
    n_jobs=-1                # use all cores
)
model.fit(train_df,train_target)

train_pred=model.predict(train_df)
val_pred=model.predict(val_df)
test_pred=model.predict(test_df)

rmse = np.sqrt(mean_squared_error(val_target, val_pred))
#print("Validation RMSE:", rmse)

rmse = np.sqrt(mean_squared_error(test_target, test_pred))
#print("Target RMSE:", rmse)

r2 = r2_score(test_target, test_pred)
mae = mean_absolute_error(test_target, test_pred)

#print("R² Score:", r2)
#print("Mean Absolute Error:", mae)

import pandas as pd
import numpy as np


def predict_count_full_input(user_input_dict, model):
    """
    user_input_dict: dictionary with keys:
        season, weather, temp, humidity, windspeed, time,
        year, month, day, weekday, am_or_pm, holidays
    model: trained RandomForest or LightGBM model
    """
    df = pd.DataFrame([user_input_dict])

    # Interaction features
    df['temp_weather'] = df['temp'] * df['weather']
    df['holiday_hour'] = df['holidays'] * df['time']

    # Cyclical encoding for time
    df['time_sin'] = np.sin(2 * np.pi * df['time'] / 24)
    df['time_cos'] = np.cos(2 * np.pi * df['time'] / 24)

    # Ensure columns are in exact order
    expected_cols = ['season', 'weather', 'temp', 'humidity', 'windspeed', 'time',
                     'year', 'month', 'day', 'weekday', 'am_or_pm', 'holidays',
                     'temp_weather', 'holiday_hour', 'time_sin', 'time_cos']

    df = df[expected_cols]

    # Predict
    pred = model.predict(df)
    return pred[0]

check='yes'


while check=='yes':

    user_input = {
        'season': int(input('Enter season (1-4): ')),
        'weather': int(input('Enter weather (1-4): ')),
        'temp': float(input('Enter temperature: ')),
        'humidity': float(input('Enter humidity: ')),
        'windspeed': float(input('Enter windspeed: ')),
        'time': int(input('Enter hour (0-23): ')),
        'year': int(input('Enter year: ')),
        'month': int(input('Enter month (1-12): ')),
        'day': int(input('Enter day (1-31): ')),
        'weekday': int(input('Enter weekday (Monday=0): ')),
        'am_or_pm': int(input('Enter AM/PM (0=AM, 1=PM): ')),
        'holidays': int(input('Enter holiday (0=No, 1=Yes): '))
    }
    # Use the function that matches training features
    predicted_count = predict_count_full_input(user_input, model)  # or rf
    print("Predicted count:", predicted_count)
    check = input('Do you Wanna Check The Count Of Rides For A Particular Hour?, yes/no')






