import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Step 1: Load the dataset
file_path_mak = "C:/Users/Mohd Arbaaz khan/OneDrive/Desktop/titanic.csv"
df_mak = pd.read_csv("C:/Users/Mohd Arbaaz khan/OneDrive/Desktop/titanic.csv")

# Step 2: Handling missing values and outliers
# Fill missing values in 'Age' with the median
df_mak['Age'].fillna(df_mak['Age'].median(), inplace=True)

# Clip outliers in 'Fare' to a reasonable upper limit (e.g., 95th percentile value)
fare_upper_limit_mak = df_mak['Fare'].quantile(0.95)
df_mak['Fare'] = np.clip(df_mak['Fare'], a_max=fare_upper_limit_mak, a_min=None)

# Step 3: Feature Engineering
# Create 'FamilySize' by combining 'SibSp' and 'Parch'
df_mak['FamilySize'] = df_mak['SibSp'] + df_mak['Parch'] + 1

# Extract titles from 'Name' and create 'Title' feature
df_mak['Title'] = df_mak['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Group rare titles into 'Rare' and map other titles to specific groups
df_mak['Title'] = df_mak['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df_mak['Title'] = df_mak['Title'].replace('Mlle', 'Miss')
df_mak['Title'] = df_mak['Title'].replace('Ms', 'Miss')
df_mak['Title'] = df_mak['Title'].replace('Mme', 'Mrs')

# Step 4: Drop unnecessary columns (Optional)
# Drop columns that are not needed for modeling.
df_mak.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1, inplace=True)

# Step 5: Data Transformation
# Use ColumnTransformer to apply different preprocessing to different columns.
# For this example, we'll use one-hot encoding for categorical variables 'Sex' and 'Embarked'.
# For numerical features, we'll use StandardScaler.
numerical_features_mak = ['Age', 'Fare', 'FamilySize']
numerical_transformer_mak = Pipeline(steps=[('scaler', StandardScaler())])

categorical_features_mak = ['Sex', 'Embarked']
categorical_transformer_mak = Pipeline(steps=[('onehot', OneHotEncoder())])

preprocessor_mak = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer_mak, numerical_features_mak),
        ('cat', categorical_transformer_mak, categorical_features_mak)
    ])

# Apply the preprocessing to the dataframe
df_transformed_mak = preprocessor_mak.fit_transform(df_mak)
print("Preprocessed Dataset:")
print(df_transformed_mak[:5])
