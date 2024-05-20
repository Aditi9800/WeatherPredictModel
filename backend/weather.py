import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sb
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings("ignore")

# Load the dataset
df = pd.read_csv('./data/flight_data.csv')

# Data exploration and preprocessing
df['Visibility_km'] = pd.to_numeric(df['Visibility_km'], errors='coerce')

cols = ['Temperature_Celsius', 'Wind_Speed_knots']
for column in cols:
    df[column] = df[column].astype(float)

def data_transform(df):
    df = df.drop(columns=['Flight_ID'])
    df = df.dropna()
    
    def labeling(value):
        if value == "High":
            return 2
        elif value == "Medium":
            return 1
        else:
            return 0
        
    df['Level'] = df['Turbulence_Level'].apply(labeling)
    df = df.astype({'Level': 'int64'})
    
    return df

# Transform the data
train = data_transform(df.copy())

# Data visualization (optional, for verification)
plt.figure(figsize=(7, 5))
sb.countplot(data=train, x='Level')
plt.title('Level')
plt.show()

for i in cols:
    plt.figure(figsize=(7, 5))
    plt.hist(train[i], bins=20, color='blue', edgecolor='black')
    plt.xlabel(f'{i}')
    plt.title(f'{i} Histogram')
    plt.tight_layout()
    plt.show()

for i in ['Visibility_km', 'Wind_Speed_knots']:
    sb.set(style="white") 
    sb.jointplot(x='Temperature_Celsius', y=i, hue='Level', data=train, s=20)
    plt.title(f'Temperature_Celsius & {i}', pad=-10)
    plt.show()

cols = ['Temperature_Celsius', 'Visibility_km', 'Wind_Speed_knots']
sb.heatmap(train[cols].corr(), annot=True, cmap='Reds')

# Split the data into features and target variable
x = train[['Temperature_Celsius', 'Visibility_km', 'Wind_Speed_knots']]
y = train['Level']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train Linear Regression model
md_lr = LinearRegression()
md_lr.fit(x_train, y_train)
pred_lr = md_lr.predict(x_test)

# Evaluate Linear Regression model
print(f'Linear Regression:')
print(f'MAE : {mean_absolute_error(y_test, pred_lr)}')
print(f'MSE : {mean_squared_error(y_test, pred_lr)}')
print(f'RMSE : {np.sqrt(mean_squared_error(y_test, pred_lr))}')
print(f'r2_score : {r2_score(y_test, pred_lr)}')

# Train Random Forest Regressor model
md_rf = RandomForestRegressor(random_state=42)
md_rf.fit(x_train, y_train)
pred_rf = md_rf.predict(x_test)

# Evaluate Random Forest Regressor model
print(f'\nRandom Forest Regressor:')
print(f'MAE : {mean_absolute_error(y_test, pred_rf)}')
print(f'MSE : {mean_squared_error(y_test, pred_rf)}')
print(f'RMSE : {np.sqrt(mean_squared_error(y_test, pred_rf))}')
print(f'r2_score : {r2_score(y_test, pred_rf)}')

# Make a single prediction
single_test_case = np.array([[-6.0, 10.0, 48.0]])
predicted_value = md_rf.predict(single_test_case)
discrete_predicted_value = round(predicted_value[0])
print("\nPredicted value (continuous):", predicted_value)
print("Predicted value (discrete):", discrete_predicted_value)

# Save the trained model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(md_rf, file)
