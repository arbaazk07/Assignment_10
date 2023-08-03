import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Step 1: Load the dataset
file_path_mak = "C:/Users/Mohd Arbaaz khan/OneDrive/Desktop/titanic.csv"
df_mak = pd.read_csv(file_path_mak)

# Step 2: Data Exploration and Visualization

# 1. Bar chart to visualize the distribution of passengers by 'Sex'
plt.figure(figsize=(6, 4))
sns.countplot(x='Sex', data=df_mak)
plt.title("Passenger Count by Sex")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.show()

# 2. Bar chart to visualize the distribution of passengers by 'Pclass'
plt.figure(figsize=(6, 4))
sns.countplot(x='Pclass', data=df_mak)
plt.title("Passenger Count by Pclass")
plt.xlabel("Pclass")
plt.ylabel("Count")
plt.show()

# 3. Scatter plot to visualize the relationship between 'Age' and 'Fare' colored by 'Survived'
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Age', y='Fare', data=df_mak, hue='Survived', palette='coolwarm')
plt.title("Age vs. Fare (Colored by Survival)")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.legend(title="Survived", loc='upper right', labels=['Not Survived', 'Survived'])
plt.show()

# Step 3: Remove non-numeric columns before calculating the correlation matrix
numerical_features_mak = df_mak.select_dtypes(include=[int, float])
correlation_matrix_mak = numerical_features_mak.corr()

# 4. Correlation heatmap to visualize the relationships between numerical features
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_mak, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 5. Box plot to visualize the distribution of fares by passenger class
plt.figure(figsize=(8, 6))
sns.boxplot(x='Pclass', y='Fare', data=df_mak)
plt.title("Fare Distribution by Pclass")
plt.xlabel("Pclass")
plt.ylabel("Fare")
plt.show()

# 6. Violin plot to visualize the distribution of ages by sex
plt.figure(figsize=(8, 6))
sns.violinplot(x='Sex', y='Age', data=df_mak)
plt.title("Age Distribution by Sex")
plt.xlabel("Sex")
plt.ylabel("Age")
plt.show()

# 7. Interactive bar chart using Plotly to visualize the distribution of passengers by 'Survived' and 'Pclass'
fig_mak = px.bar(df_mak, x='Pclass', color='Survived', barmode='group', title='Passenger Count by Pclass and Survival',
             labels={'Pclass': 'Passenger Class', 'Survived': 'Survived', 'count': 'Count'})
fig_mak.show()
