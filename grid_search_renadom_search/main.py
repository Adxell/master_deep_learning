from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np


data = load_iris()
X, y = data.data, data.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)


#display dataset
print(f"Feature Name: {data.feature_names}")
print(f"Class Name: {data.target_names}")


param_grid = {
    'n_estimators': [50, 100, 150], 
    'max_depth': [None, 5, 10], 
    'min_samples_split': [2, 5, 10]
}


# Initialize GRID Search 
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42), 
    param_grid=param_grid, 
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

#Perform Grid Search 
grid_search.fit(X_train, y_train)

# Evaluate best model 
best_grid_model = grid_search.best_estimator_
y_pred_grid = best_grid_model.predict(X_test)
accuracy_grid = accuracy_score(y_test, y_pred_grid)


print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Grid Search Accuracy: {accuracy_grid:.4f}")



# Define 


param_dist = {
    'n_estimators': np.arange(50, 200, 10), 
    'max_depth': [None, 5, 10, 15], 
    'min_samples_split': [2, 5, 10, 20]
}

# Inizialite Random Search 
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(n_estimators=42), 
    param_distributions=param_dist, 
    n_iter=20, 
    scoring='accuracy', 
    n_jobs=-1, 
    random_state=42,
)

# Performe search 

random_search.fit(X_train, y_train)

# Evaluate best model 

best_random_model = random_search.best_estimator_
y_pred_random = best_random_model.predict(X_test)
accuracy_random = accuracy_score(y_test, y_pred_random)


print(f"Best Hyperparameters (Random Search): {random_search.best_params_}")
print(f"Random Search Accuracy: {accuracy_random:.4f}")

