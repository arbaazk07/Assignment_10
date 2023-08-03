import rpy2.robjects as mak_robjects
import matplotlib.pyplot as mak_plt
import pandas as mak_pd
from sklearn.model_selection import train_test_split as mak_train_test_split
from sklearn.linear_model import LogisticRegression as mak_LogisticRegression
from sklearn.tree import DecisionTreeClassifier as mak_DecisionTreeClassifier
from sklearn.metrics import accuracy_score as mak_accuracy_score, classification_report as mak_classification_report, confusion_matrix as mak_confusion_matrix
import seaborn as mak_sns

# Load the Titanic dataset in R and convert it into an R dataframe
mak_r = mak_robjects.r
mak_r('titanic_data <- read.csv("C:/Users/Mohd Arbaaz khan/OneDrive/Desktop/titanic.csv")')

# Fetch the Titanic dataset from R to Python
titanic_data = mak_robjects.globalenv['titanic_data']

# Convert the R dataframe to a Pandas dataframe
df = mak_pd.DataFrame(dict(titanic_data.items()))

# Print the head of the dataset
result_head = mak_r('head(titanic_data)')
print(result_head)

# EDA Task 2
# print("Summary statistics of numerical features:")
print(df.describe())

# Check the data types and missing values
print("Data types and missing values:")
print(df.info())

# Visualize the distribution of numerical features using histograms
numerical_features = ['Age', 'Fare', 'SibSp', 'Parch']
mak_plt.figure(figsize=(12, 8))
for i, feature in enumerate(numerical_features):
    mak_plt.subplot(2, 2, i + 1)
    mak_sns.histplot(data=df, x=feature, kde=True, bins=20, color='skyblue')
    mak_plt.title(f'Histogram of {feature}')
mak_plt.tight_layout()
mak_plt.show()

# Plot the bar plot for Passenger Class (Pclass) distribution and save it as "barplot.png"
mak_r('png("barplot.png")')
mak_r('barplot(table(titanic_data$Pclass), main="Passenger Class Distribution", xlab="Passenger Class", ylab="Count", col="orange")')
mak_r['dev.off']()

# Load and display the saved bar plot using matplotlib in Python
img = mak_plt.imread("barplot.png")
mak_plt.imshow(img)
mak_plt.axis("off")
mak_plt.show()

# Preprocessing 3
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = mak_pd.get_dummies(df, columns=['Embarked'], prefix='Embarked')

# Drop unnecessary columns for modeling
X = df.drop(columns=['PassengerId', 'Name', 'Survived', 'Cabin', 'Ticket', 'Title'])
y = df['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = mak_train_test_split(X, y, test_size=0.2, random_state=42)

# Implement Logistic Regression
logreg_model = mak_LogisticRegression(max_iter=1000)
logreg_model.fit(X_train, y_train)
y_pred_logreg = logreg_model.predict(X_test)

# Implement Decision Tree Classifier
dt_model = mak_DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Evaluate the models
print("Logistic Regression Model:")
print("Accuracy:", mak_accuracy_score(y_test, y_pred_logreg))
print("Classification Report:")
print(mak_classification_report(y_test, y_pred_logreg))
print("Confusion Matrix:")
print(mak_confusion_matrix(y_test, y_pred_logreg))

print("\nDecision Tree Classifier:")
print("Accuracy:", mak_accuracy_score(y_test, y_pred_dt))
print("Classification Report:")
print(mak_classification_report(y_test, y_pred_dt))
print("Confusion Matrix:")
print(mak_confusion_matrix(y_test, y_pred_dt))

# Scatter Plot for 'Age' vs 'Fare'
mak_plt.scatter(df['Age'], df['Fare'], c=df['Survived'], cmap='viridis')
mak_plt.xlabel('Age')
mak_plt.ylabel('Fare')
mak_plt.title('Scatter Plot: Age vs Fare (Color by Survival)')
mak_plt.colorbar(label='Survived')
mak_plt.show()
