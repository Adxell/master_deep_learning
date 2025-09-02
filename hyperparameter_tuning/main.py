from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import optuna


data = load_breast_cancer()

X, y = data.data, data.target


#Split data into traing and test
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Traning data shape before scaler", X_train.shape)
print("Test data shape before scaler", X_test.shape)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


print("Traing data shape: ", X_train.shape)
print("Test data shape: ", X_test.shape)


# Train a base line XGBoost model
baseline_model = XGBClassifier(eval_metrics='logloss', random_state=42)
baseline_model.fit(X_train, y_train)


baseline_preds = baseline_model.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_preds)
print(f"Baseline XGBoost ACcuracy: {baseline_accuracy:.4f}")

#Define the objeciutive function for optuna 

def objective(trial): 
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500), 
        'max_depth': trial.suggest_int('max_depth', 3, 100), 
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3), 
        'subsample': trial.suggest_float('subsample', 0.6, 1.0), 
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
    }

    # Train XGBoost model with suggested params 
    model = XGBClassifier(eval_metric='logloss', ramdon_state=42, **params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

# Create optuna study 

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Best hypperparameters 
print("Best HyperParameters: ", study.best_params)
print("Best HyperParameters: ", study.best_value)


param_grid = {
    'n_estimators': [100, 200, 300], 
    'max_depth': [3, 5, 7], 
    'learing_rate': [0.01, 0.1, 0.2], 
    'subsample': [0.6, 0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=XGBClassifier(eval_metrics='logloss', random_state=42), 
    param_grid=param_grid, 
    scoring='accuracy', 
    cv=3, 
    verbose=1
)