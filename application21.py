from ast import increment_lineno
from statistics import LinearRegression
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set(style='whitegrid')

print('Import and setup completed successfully.')

file_path = ''

file_path = r'C:\Users\Donte Patton\Downloads\dataset_2191_sleep.csv'
df = pd.read_csv(file_path, encoding='ascii', delimiter=',')

print('Dataset loaded successfull. Showing first few rows:')
print(df.head())

print('Dataset Info:')
df.info()

print('\nMissing values in each column:')
print(df.isnull().sum())

df.dropna(inplace=True)
print('\nDataframe shape after dropping missing values:', df.shape)

# Removed Year conversion as the column doesn't exist in the dataset

print('\nData types after conversion:')
print(df.dtypes)

numeric_df = df.select_dtypes(include=[np.number])

if numeric_df.shape[1] >= 4:
    plt.figure(figsize=(12, 10))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Numeric Variables')
    plt.show()
else:
    print('Not enough numeric columns for a correlation heatmap.')

# Using available numeric columns for pairplot
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
if len(numeric_cols) > 1:
    sns.pairplot(df[numeric_cols])
    plt.suptitle('Pair Plot of Numeric Features', y=1.02)
    plt.show()
else:
    print('Not enough numeric columns for pair plot.')

# Plotting distribution of body_weight instead of CO2
plt.figure(figsize=(8, 6))
sns.histplot(df['body_weight'], kde=True, bins=30)
plt.title('Distribution of Body Weight')
plt.xlabel('Body Weight (kg)')
plt.ylabel('Frequency')
plt.show()

# Plotting mean body weight by predation index
plt.figure(figsize=(10, 6))
body_weight_by_predation = df.groupby('predation_index')['body_weight'].mean().reset_index()
sns.barplot(x='predation_index', y='body_weight', data=body_weight_by_predation, palette='viridis')
plt.title('Average Body Weight by Predation Index')
plt.xlabel('Predation Index')
plt.ylabel('Average Body Weight (kg)')
plt.show()

# Create a count plot for predation_index instead of Emissions Category
plt.figure(figsize=(8, 6))
sns.countplot(x='predation_index', data=df, palette='Set2')
plt.title('Count of Records by Predation Index')
plt.xlabel('Predation Index')
plt.ylabel('Count')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Update features to use existing numeric columns
features = ['body_weight', 'brain_weight', 'predation_index', 'sleep_exposure_index', 'danger_index']

# Convert string columns to numeric where needed
model_df = df.copy()

# Convert total_sleep to numeric (it's currently an object/string)
model_df['total_sleep'] = pd.to_numeric(model_df['total_sleep'], errors='coerce')

# Drop any rows with missing values
model_df = model_df.dropna()

# Use available numeric features for prediction
# We'll predict 'total_sleep' using other numeric features
X = model_df[['body_weight', 'brain_weight', 'predation_index', 'sleep_exposure_index', 'danger_index']]
y = model_df['total_sleep']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Training set shape:', X_train.shape)
print('Testing set shape:', X_test.shape)

# Train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions
y_pred = regressor.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'R^2 score for the predictor: {r2:.3f}')
print(f'RMSE for the predictor: {rmse:.3f}')

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual CO2')
plt.ylabel('Predicted CO2')
plt.title('Actual vs Predicted CO2 Emissions')
plt.show()

