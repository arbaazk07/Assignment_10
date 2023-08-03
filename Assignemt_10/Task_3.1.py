import pandas as pd
import numpy as np

# Step 1: Load the dataset
file_path = "C:/Users/Mohd Arbaaz khan/OneDrive/Desktop/titanic.csv"
df_mak = pd.read_csv("C:/Users/Mohd Arbaaz khan/OneDrive/Desktop/titanic.csv")

# Step 2: Handling missing values
# For this example, we'll fill the missing values in 'Age' with the mean and 'Embarked' with the mode.
df_mak['Age'].fillna(df_mak['Age'].mean(), inplace=True)
df_mak['Embarked'].fillna(df_mak['Embarked'].mode()[0], inplace=True)

# Step 3: Encoding categorical variables
# Convert 'Sex' and 'Embarked' columns into numerical values using one-hot encoding.
df_mak = pd.get_dummies(df_mak, columns=['Sex', 'Embarked'], drop_first=True)

# Step 4: Feature Engineering 
# We can create new features from existing ones. For example, we can combine 'SibSp' and 'Parch' to create 'FamilySize'.
df_mak['FamilySize'] = df_mak['SibSp'] + df_mak['Parch'] + 1

# Step 5: Drop unnecessary columns 
# If certain columns are not needed, we can drop them.
df_mak.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1, inplace=True)


# Display the preprocessed dataset
print("Preprocessed Dataset:")
print(df_mak.head())


### mak is my initials ###