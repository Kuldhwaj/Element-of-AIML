# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data Acquisition
# For this example, we'll create a DataFrame manually. In practice, you would load a dataset from a file or a URL.
data = {
    'Age': [34, 34, 37, 30, 30, 27, 34, 34, 30, 36, 34, 28, 35, 34, 34, 37, 30, 30],
    'FrequentFlyer': ['No', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'No', 'No Record', 'Yes', 'No', 'Yes', 'No', 'Yes'],
    'AnnualIncomeClass': ['Middle Income', 'Low Income', 'Middle Income', 'Middle Income', 'Low Income', 'High Income', 
                          'Middle Income', 'Low Income', 'Low Income', 'High Income', 'Low Income', 'Middle Income', 
                          'Middle Income', 'Low Income', 'Middle Income', 'Low Income', 'Low Income', 'High Income'],
    'ServicesOpted': [6, 5, 3, 2, 1, 1, 4, 2, 3, 1, 1, 2, 1, 4, 5, 6, 1, 1],
    'AccountSyncedToSocialMedia': ['No', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes'],
    'BookedHotelOrNot': ['Yes', 'No', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No'],
    'Target': [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
}

# Create DataFrame
df = pd.DataFrame(data)

# 2. Data Import and Export Using Pandas
# Display the first few rows of the DataFrame
print("DataFrame Head:")
print(df.head())

# Get the structure of the DataFrame
print("\nDataFrame Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 3. Exploratory Data Analysis (EDA)
# Statistical summary
print("\nStatistical Summary:")
print(df.describe(include='all'))

# Visualizations
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Target')
plt.title('Count of Target Variable')
plt.xlabel('Target')
plt.ylabel('Count')
plt.show()

# Visualizing age distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=10, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 4. Handling Missing Values and Outliers
# Replace 'No Record' with NaN for handling missing values
df.replace('No Record', np.nan, inplace=True)

# Check for missing values again
print("\nMissing Values After Replacement:")
print(df.isnull().sum())

# Fill missing values in 'FrequentFlyer' with the mode
df['FrequentFlyer'].fillna(df['FrequentFlyer'].mode()[0], inplace=True)

# Detecting outliers using IQR method for 'ServicesOpted'
Q1 = df['ServicesOpted'].quantile(0.25)
Q3 = df['ServicesOpted'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identifying outliers
outliers = df[(df['ServicesOpted'] < lower_bound) | (df['ServicesOpted'] > upper_bound)]
print("\nOutliers in ServicesOpted:")
print(outliers)

# Removing outliers from the dataset
df = df[~((df['ServicesOpted'] < lower_bound))]