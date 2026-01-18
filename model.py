### Predicting Whether a Passenger Survived the Titanic Disaster

#importing neccessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
import joblib

# Load the dataset
data = pd.read_csv('Data/train.csv')

# previewing the first five rows of the dataset
print(data.head())

## Data Preprocessing
# Dropping unnecessary columns not useful for prediction
data = data.drop(columns=['Name', 'Ticket', 'Cabin'], axis=1)

# Handling missing values
data['Age'] = data['Age'].fillna(data['Age'].median())

#filling missing values in "Embarked" column with the mode
if 'Embarked' in data.columns:
    mode_value = data['Embarked'].mode()[0]
    data['Embarked'] = data['Embarked'].fillna(mode_value)
else:
    print("Column 'Embarked' not found in the dataset.")

# Converting categorical variables into dummy variables
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

#scaling the numerical features
scaler = StandardScaler()
numerical_features = ['Age', 'Fare', 'SibSp', 'Parch', 'Pclass']
data[numerical_features] = scaler.fit_transform(data[numerical_features])


# split the data into features and target variable
X = data.drop(columns=['Survived'], axis=1)
y = data['Survived']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the Logistic Regression model
#initializing the Logistic Regression model
model = LogisticRegression()

#training the model
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

#printing evaluation metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')  
print(f'F1 Score: {f1:.2f}')

# Displaying the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])   
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# plotting the ROC curve to visualize the classification performance.
# calculate the ROC curve to visualize the classification performance
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)

# plotting the ROC curve
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue',  label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# === Export trained Logistic Regression model ===

joblib.dump(model, r'model/logistic_regression_model.joblib')
print('Model saved to: model/logistic_regression_model.joblib')