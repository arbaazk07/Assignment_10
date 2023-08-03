import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Rest of the code remains the same...

# Step 1: Load the dataset
file_path_mak = "C:/Users/Mohd Arbaaz khan/OneDrive/Desktop/titanic.csv"
df_mak = pd.read_csv("C:/Users/Mohd Arbaaz khan/OneDrive/Desktop/titanic.csv")

# Step 2: Preprocessing (You can reuse the preprocessing steps from the previous code)

# Check if the 'Title' column exists before dropping it
if 'Title' in df_mak.columns:
    df_mak.drop('Title', axis=1, inplace=True)

# Step 3: Split the data into training and testing sets
X_mak = df_mak.drop('Survived', axis=1)
y_mak = df_mak['Survived']
X_train_mak, X_test_mak, y_train_mak, y_test_mak = train_test_split(X_mak, y_mak, test_size=0.2, random_state=42)

# Step 4: Feature Scaling (not required for all models, but we'll use it for SVM)
# Drop non-numeric columns before scaling
X_train_numeric_mak = X_train_mak.select_dtypes(include=[int, float])
X_test_numeric_mak = X_test_mak.select_dtypes(include=[int, float])

# Handle missing values
imputer_mak = SimpleImputer(strategy='mean')
X_train_imputed_mak = imputer_mak.fit_transform(X_train_numeric_mak)
X_test_imputed_mak = imputer_mak.transform(X_test_numeric_mak)

scaler_mak = StandardScaler()
X_train_scaled_mak = scaler_mak.fit_transform(X_train_imputed_mak)
X_test_scaled_mak = scaler_mak.transform(X_test_imputed_mak)

# Step 5: Implement the models
# Model 1: Support Vector Machine (SVM)
svm_model_mak = SVC(kernel='rbf', random_state=42)
svm_model_mak.fit(X_train_scaled_mak, y_train_mak)

# Model 2: Random Forest
rf_model_mak = RandomForestClassifier(random_state=42)
rf_model_mak.fit(X_train_imputed_mak, y_train_mak) # Use X_train_imputed_mak instead of X_train_numeric_mak

# Step 6: Evaluate the models
def evaluate_model_mak(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

svm_accuracy_mak, svm_precision_mak, svm_recall_mak, svm_f1_mak = evaluate_model_mak(svm_model_mak, X_test_scaled_mak, y_test_mak)
rf_accuracy_mak, rf_precision_mak, rf_recall_mak, rf_f1_mak = evaluate_model_mak(rf_model_mak, X_test_imputed_mak, y_test_mak) # Use X_test_imputed_mak instead of X_test_numeric_mak

# Step 7: Compare the performance
print("Support Vector Machine (SVM) Performance:")
print(f"Accuracy: {svm_accuracy_mak:.2f}")
print(f"Precision: {svm_precision_mak:.2f}")
print(f"Recall: {svm_recall_mak:.2f}")
print(f"F1 Score: {svm_f1_mak:.2f}")

print("\nRandom Forest Performance:")
print(f"Accuracy: {rf_accuracy_mak:.2f}")
print(f"Precision: {rf_precision_mak:.2f}")
print(f"Recall: {rf_recall_mak:.2f}")
print(f"F1 Score: {rf_f1_mak:.2f}")
