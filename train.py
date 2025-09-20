import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 1️⃣ Load data
df = pd.read_csv("insurance.csv")
print("Initial data:")
print(df.head(), "\n")

# 2️⃣ Encode categorical columns
categorical_cols = ['smoker', 'sex', 'region']
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

print("After Label Encoding:")
print(df.head(), "\n")

# 3️⃣ Train-test split
X = df.drop(columns='smoker')
y = df['smoker']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)
print('X_train size: {}, X_test size: {}'.format(X_train.shape, X_test.shape), "\n")

# 4️⃣ Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 5️⃣ Function to find best model using GridSearchCV
def find_best_model(X, y):
    models = {
        'logistic_regression': {
            'model': LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=500),
            'parameters': {'C': [1, 5, 10]}
        },
        'decision_tree': {
            'model': DecisionTreeClassifier(splitter='best', random_state=0),
            'parameters': {'criterion': ['gini', 'entropy'], 'max_depth': [5, 10]}
        },
        'random_forest': {
            'model': RandomForestClassifier(criterion='gini', random_state=0),
            'parameters': {'n_estimators': [10, 20, 50]}
        },
        'svm': {
            'model': SVC(gamma='auto'),
            'parameters': {'C': [1, 10], 'kernel': ['rbf','linear']}
        }
    }
    
    scores = []
    cv_shuffle = ShuffleSplit(n_splits=3, test_size=0.20, random_state=0)  # faster for testing

    for model_name, model_params in models.items():
        print(f"Training {model_name}...")
        gs = GridSearchCV(
            model_params['model'],
            model_params['parameters'],
            cv=cv_shuffle,
            return_train_score=False
        )
        gs.fit(X, y)
        scores.append({
            'model': model_name,
            'best_parameters': gs.best_params_,
            'score': gs.best_score_
        })
    
    return pd.DataFrame(scores, columns=['model','best_parameters','score'])

# 6️⃣ Run and print best model results
print("Finding best model using GridSearchCV...\n")
result = find_best_model(X_train, y_train)
print("\nBest models and their scores:\n", result, "\n")

# 7️⃣ Using cross_val_score for RandomForest average accuracy
rf = RandomForestClassifier(n_estimators=20, random_state=0)
scores = cross_val_score(rf, X_train, y_train, cv=5)
print('Average Accuracy of RandomForest (5-fold CV): {:.2f}%'.format(scores.mean()*100), "\n")

# 8️⃣ Train Random Forest on training data
rf.fit(X_train, y_train)

# 9️⃣ Evaluate on test set
y_pred = rf.predict(X_test)
cm_test = confusion_matrix(y_test, y_pred)
print("Confusion Matrix (Test set):\n", cm_test)
print("Accuracy on test set: {:.2f}%".format(accuracy_score(y_test, y_pred)*100))
print("Classification Report (Test set):\n", classification_report(y_test, y_pred))

# 10️⃣ Evaluate on training set
y_train_pred = rf.predict(X_train)
cm_train = confusion_matrix(y_train, y_train_pred)
print("Confusion Matrix (Training set):\n", cm_train)
print("Accuracy on training set: {:.2f}%".format(accuracy_score(y_train, y_train_pred)*100))
print("Classification Report (Training set):\n", classification_report(y_train, y_train_pred))     



import joblib

# Model save karo
joblib.dump(rf, "random_forest_model.joblib")

# Agar scaler bhi use kiya hai, usko bhi save kar do
joblib.dump(sc, "scaler.joblib")

print("✅ Model and scaler saved!")


import joblib

# Load saved model
rf = joblib.load("random_forest_model.joblib")

# Load saved scaler
sc = joblib.load("scaler.joblib")
