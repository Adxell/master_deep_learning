from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
import numpy as np

# Load dataset
data = load_iris()

X, y = data.data, data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print('Dataset Loaded and Split Seccessfully')

# Define parameters grid
parameters_grid = {
    'n_estimators': [50, 100, 150], 
    'learning_rate': [0.01, 0.1, 0.2], 
    'max_depth': [3, 5, 7], 
}

# Initialize GridSearchCV
grid_serach = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42), 
    param_grid=parameters_grid, 
    scoring='accuracy', 
    cv=5, 
    n_jobs=-1
)

grid_serach.fit(X_train, y_train)

# Get best paramters and score 
best_params_grid = grid_serach.best_params_
best_score_grid = grid_serach.best_score_


print(f"Best paramters (GRidSearchCV): {best_params_grid}")
print(f"Best Scoring (GRidSearchCV): {best_score_grid}")

best_grid_model = grid_serach.best_estimator_

y_pred_grid = best_grid_model.predict(X_test)

accuracy_grid = accuracy_score(y_test, y_pred_grid)

print(f"Test accuracy (GridSearchCV): {accuracy_grid:.4f}")
print("\n Classificacion report: \n", classification_report(y_test, y_pred_grid))